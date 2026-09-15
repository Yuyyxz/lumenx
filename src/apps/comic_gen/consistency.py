"""跨镜头一致性 — Seedance r2v 锁定行提示词 + @图片N 别名生成器。

来源: T-V2 调研 §3.1 #2（侠客漫 seedance-final-video-prompt 的锁定行/别名思想,
license=Other —— 只重写实现, 不 vendor 代码）。纯函数, 无 I/O。

五分型锁定行（每张参考图一条, 告诉 r2v 模型"这张图锁什么、不锁什么"）:
- character_identity 角色身份: 全程 100% 锁定五官/脸型/发型/年龄感/体态,
  禁止换脸换发型变年龄 —— 跨镜头同脸的核心
- character_outfit 服装变装: 只锁服装体态, 脸以身份参考图为准（换装不破相）
- faceless_reference 无脸参考: 禁止读取五官/肖像（背影/剪影/道具图防污染）
- scene 场景: 锁 space/lighting/color/camera-axis（空间/光照/色彩/轴线）
- group 群像: 不要求个体同脸（人群/背景路人）

@图片N 别名: build_reference_aliases 把资产名映射成 Seedance 可寻址的
@图片1..N（按传入顺序）。别名表随 task 持久化便于审计（上游职责）。

接入点: prompt_assembly.assemble_r2v_prompt_with_locks（r2v 提交前把锁定行
拼进 prompt）。reference 分型元数据到齐后, pipeline 只需一行调用。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Tuple


class ReferenceLockKind(str, Enum):
    CHARACTER_IDENTITY = "character_identity"
    CHARACTER_OUTFIT = "character_outfit"
    FACELESS = "faceless_reference"
    SCENE = "scene"
    GROUP = "group"


@dataclass(frozen=True)
class ReferenceLock:
    """一张参考图 + 它的分型。description 为可选英文锁定描述（禁人名）。"""
    name: str
    kind: ReferenceLockKind
    description: str = ""
    alias: str = field(default="")  # 由 build_reference_lock_lines 自动填充


# 五分型锁定行模板（重写实现; 锁定语义对齐 T-V2 §2.2/§3.1#2 的口径）
LOCK_LINE_TEMPLATES: Dict[ReferenceLockKind, str] = {
    ReferenceLockKind.CHARACTER_IDENTITY: (
        "{alias}（{name}｜角色身份参考）全程 100% 锁定该角色：五官、脸型、发型、"
        "年龄感与体态一律以本图为准，任何镜头不得换脸、换发型或变年龄。"
    ),
    ReferenceLockKind.CHARACTER_OUTFIT: (
        "{alias}（{name}｜服装变装参考）仅锁定本图中的服装与体态；"
        "脸部仍以该角色的身份参考图为准。"
    ),
    ReferenceLockKind.FACELESS: (
        "{alias}（{name}｜无脸参考）仅作构图、氛围与物件参考；"
        "禁止读取本图中的脸、五官与肖像信息。"
    ),
    ReferenceLockKind.SCENE: (
        "{alias}（{name}｜场景参考）锁定空间结构、光照、色彩与镜头轴线；"
        "镜头运动保持同一轴线与透视，不得改变场景布局。"
    ),
    ReferenceLockKind.GROUP: (
        "{alias}（{name}｜群像参考）作为整体构图与密度参考；"
        "不要求画面中每个个体与参考图同脸。"
    ),
}


def build_reference_aliases(names: Iterable[str]) -> Dict[str, str]:
    """资产名 → @图片N 别名（N 从 1 起, 按传入顺序）。

    同名资产重复出现时后者覆盖前者（同一资产两张图应去重后再传入）。
    """
    return {name: f"@图片{idx}" for idx, name in enumerate(names, start=1)}


def build_reference_lock_line(ref: ReferenceLock, alias: str) -> str:
    """单条锁定行 = 分型模板 + 可选外观锁定描述。"""
    template = LOCK_LINE_TEMPLATES[ref.kind]
    line = template.format(alias=alias, name=ref.name)
    if ref.description:
        line += f"外观锁定：{ref.description}"
    return line


def build_reference_lock_lines(refs: List[ReferenceLock]) -> List[str]:
    """每个参考一条锁定行, alias 按 refs 顺序自动分配为 @图片1..N。"""
    lines: List[str] = []
    for idx, ref in enumerate(refs, start=1):
        alias = f"@图片{idx}"
        lines.append(build_reference_lock_line(ref, alias))
    return lines


def assemble_r2v_prompt(
    base_prompt: str,
    refs: List[ReferenceLock],
) -> Tuple[str, Dict[str, str]]:
    """R2V 提交前的最终 prompt：基础画面描述 + 锁定行块。

    返回 (prompt, alias_map)。alias_map 为 资产名→@图片N（随 task 持久化便于审计）。
    没有参考时原样返回基础 prompt（空别名表）。
    """
    alias_map = build_reference_aliases([ref.name for ref in refs])
    if not refs:
        return base_prompt, alias_map
    lock_block = "参考图锁定：\n" + "\n".join(build_reference_lock_lines(refs))
    clean = (base_prompt or "").rstrip()
    prompt = f"{clean}\n\n{lock_block}" if clean else lock_block
    return prompt, alias_map
