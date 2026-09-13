"""Kling video generation model adapter.

API: https://api-beijing.klingai.com (可灵 API 2.0, 路径式模型端点)
Auth: Bearer <API Key> (开放平台单 key, 不再用 AK/SK JWT)
Models: kling-3.0 / kling-3.0-turbo (模型 ID 位于路径中, 新版设计标准)

API 2.0 迁移要点 (2026-08 实测):
- 旧版: POST /v1/videos/image2video, model_name 在 body, AK/SK 签 JWT
- 新版: POST /image-to-video/{model_id}, 模型在路径, Bearer 单 key
- 新版请求体: contents[] / settings{} / options{} 三层结构
- 新版轮询: GET /tasks?task_ids=xxx (批量), 返回 data[] 数组

T-B2 可靠性改造 (不改 2.0 协议语义):
- 裸 RuntimeError → 结构化 KlingError(code/message/request_id/status_code/
  retryable/category)，字段口径参照官方 ComfyUI KlingAPIError
- 官方错误码分类: 1302/1303/5000/5001 + 网络类可重试; 1100-1103 账户类
  (欠费/资源包耗尽/无权访问) 一律不可重试
- 凭证校验 check_credentials() 走 GET /account/costs（不再烧资源包验证）
- 轮询退避: 首查 2s，delay=min(delay*1.5, 10)+jitter（参照 framepipe 公式）;
  提交/轮询遇 1302/1303 指数退避后重试
"""

import logging
import os
import random
import time
from typing import Dict, Any, Tuple, Optional

import requests

from .base import VideoGenModel
from ..utils.endpoints import get_provider_base_url

logger = logging.getLogger(__name__)


class KlingError(RuntimeError):
    """可灵 API 结构化异常。

    request_id 是找官方客服排障的唯一凭据，务必透传到日志与任务状态卡。
    retryable 语义: 只有官方限频(1302)/超并发(1303)/服务端(5000/5001)与
    网络类错误才可重试；账户类 1100-1103（欠费/资源包耗尽等）重试无意义。
    """

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        message_en: Optional[str] = None,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        retryable: bool = False,
        category: str = "unknown",
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.message_en = message_en
        self.request_id = request_id
        self.status_code = status_code
        self.retryable = retryable
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "message_en": self.message_en,
            "request_id": self.request_id,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "category": self.category,
        }


# 官方错误码 → (category, retryable)
# 依据官方错误码表（survey V6 §2.2，官方 ComfyUI 节点 / open-connector /
# aself101 文档三方交叉印证）:
#   0 成功; 1000-1004 鉴权; 1100 账户异常 / 1101 欠费 / 1102 资源包耗尽或过期
#   / 1103 无权访问（重试无意义）; 1200-1203 参数; 1300/1301/1304 政策;
#   1302 限频 / 1303 超并发（429，可退避重试）; 5000/5001 服务端
_KLING_ERROR_TABLE: Dict[int, Tuple[str, bool]] = {}
for _c in (1000, 1001, 1002, 1003, 1004):
    _KLING_ERROR_TABLE[_c] = ("auth", False)
for _c in (1100, 1101, 1102, 1103):
    _KLING_ERROR_TABLE[_c] = ("account", False)
for _c in (1200, 1201, 1202, 1203):
    _KLING_ERROR_TABLE[_c] = ("params", False)
for _c in (1300, 1301, 1304):
    _KLING_ERROR_TABLE[_c] = ("policy", False)
for _c in (1302, 1303):
    _KLING_ERROR_TABLE[_c] = ("rate_limit", True)
for _c in (5000, 5001):
    _KLING_ERROR_TABLE[_c] = ("server", True)

# HTTP 层可重试状态码（网关/服务端瞬断）
_RETRYABLE_HTTP_STATUS = {429, 502, 503, 504}

# 提交/轮询遇到这些官方码时做指数退避后重试
_RETRYABLE_KLING_CODES = {1302, 1303, 5000, 5001}

# 轮询终态（v2.0 还有 cancelled / expired，V6 差距清单补齐）
_TERMINAL_TASK_STATUSES = {"failed", "cancelled", "expired"}


def classify_kling_error(code: Optional[int] = None, status_code: Optional[int] = None) -> Tuple[str, bool]:
    """错误码 → (category, retryable)。官方 code 优先，其次按 HTTP 状态。"""
    if code is not None:
        if code == 0:
            return ("success", False)
        if code in _KLING_ERROR_TABLE:
            return _KLING_ERROR_TABLE[code]
    if status_code is not None:
        if status_code in _RETRYABLE_HTTP_STATUS:
            return ("http", True)
        if 500 <= status_code < 600:
            return ("http", True)
        if 400 <= status_code < 500:
            return ("http", False)
    return ("unknown", False)


def _wrap_request_exception(e: requests.RequestException, what: str) -> KlingError:
    """网络层异常（超时/连接失败）→ 可重试 KlingError。带 HTTP 响应的按状态码分类。"""
    status_code = None
    response = getattr(e, "response", None)
    if response is not None:
        status_code = response.status_code
    if status_code is None:
        retryable = True  # 纯网络错误（超时/断连）
    else:
        retryable = classify_kling_error(status_code=status_code)[1]
    return KlingError(
        f"Kling {what} 网络错误: {e}",
        status_code=status_code,
        retryable=retryable,
        category="network",
    )


def _kling_error_from_response(response: requests.Response, what: str) -> KlingError:
    """非 200 HTTP 响应 → KlingError。优先从响应体取官方 code/request_id。"""
    body_code = None
    body_message = None
    request_id = None
    try:
        payload = response.json()
        body_code = payload.get("code")
        body_message = payload.get("message")
        request_id = (
            payload.get("request_id")
            or response.headers.get("X-Kling-Request-Id")
        )
    except (ValueError, AttributeError):
        pass
    category, retryable = classify_kling_error(code=body_code, status_code=response.status_code)
    return KlingError(
        f"Kling {what} HTTP {response.status_code}: {response.text[:300]}",
        code=body_code,
        message_en=body_message,
        request_id=request_id,
        status_code=response.status_code,
        retryable=retryable,
        category=category,
    )


def _backoff_delay(attempt: int, base: float = 1.0, cap: float = 10.0) -> float:
    """指数退避 + 随机抖动。attempt 从 1 开始。"""
    delay = min(cap, base * (2 ** (attempt - 1)))
    return delay + random.uniform(0, 0.5)


class KlingModel(VideoGenModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # API 2.0: 开放平台单 key, Bearer 直用
        self.api_key = config.get("api_key") or os.getenv("KLING_API_KEY", "")
        # 模型 ID (路径式, 新版标准): kling-3.0 / kling-3.0-turbo / kling-3.0-omni
        self.model_name = config.get("params", {}).get("model_name", "kling-3.0")

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def check_credentials(self) -> Dict[str, Any]:
        """凭证校验：GET /account/costs（消费查询端点，不烧资源包）。

        来源：open-connector validateKlingCredential（Apache-2.0）——
        用 start_time/end_time 查询消费，响应可解析且有 data 即 key 有效。
        返回 data 字段（含消费记录，可用于成本标定）；无效凭证/网络问题
        抛 KlingError（分类语义与生成路径一致）。
        """
        if not self.api_key:
            raise KlingError(
                "KLING_API_KEY 未配置",
                retryable=False,
                category="auth",
            )
        base_url = get_provider_base_url("KLING")
        now_ms = int(time.time() * 1000)
        url = f"{base_url}/account/costs"
        try:
            resp = requests.get(
                url,
                headers=self._auth_headers(),
                params={"start_time": now_ms - 60_000, "end_time": now_ms},
                timeout=30,
            )
        except requests.RequestException as e:
            raise _wrap_request_exception(e, "credentials check") from e
        if resp.status_code != 200:
            raise _kling_error_from_response(resp, "credentials check")
        try:
            payload = resp.json()
        except ValueError as e:
            raise KlingError(
                f"Kling credentials check 响应非 JSON: {resp.text[:200]}",
                status_code=resp.status_code,
                retryable=False,
                category="unknown",
            ) from e
        if payload.get("code") not in (0, None):
            category, retryable = classify_kling_error(code=payload.get("code"), status_code=resp.status_code)
            raise KlingError(
                f"Kling credentials check 失败 (code {payload.get('code')}): {payload.get('message')}",
                code=payload.get("code"),
                message_en=payload.get("message"),
                request_id=payload.get("request_id"),
                status_code=resp.status_code,
                retryable=retryable,
                category=category,
            )
        data = payload.get("data")
        if data is None:
            raise KlingError(
                "Kling credentials check 响应缺少 data 字段",
                status_code=resp.status_code,
                retryable=False,
                category="unknown",
            )
        logger.info("[Kling] credentials OK (account/costs reachable)")
        return data

    def _resolve_image_input(self, img_url: str = None, img_path: str = None) -> str:
        """解析图片输入为 URL (新版 first_frame 需要 url)。"""
        if img_url and img_url.startswith(("http://", "https://")):
            return img_url
        if img_path and os.path.exists(img_path):
            # 本地文件需上传到可灵 CDN 或用 data URL。
            # 简化: 上传逻辑可参考 utils/oss_utils, 这里先支持远程 URL。
            raise ValueError(
                "API 2.0 需要图片 URL; 本地文件请先上传 (可灵 CDN 或对象存储)"
            )
        return img_url or ""

    def generate(self, prompt: str, output_path: str, img_url: str = None,
                 img_path: str = None, **kwargs) -> Tuple[str, float]:
        """Generate video using Kling API 2.0 (T2V or I2V)."""
        if not self.api_key:
            raise KlingError(
                "KLING_API_KEY 未配置",
                retryable=False,
                category="auth",
            )

        duration = int(kwargs.get("duration", 5))
        # 参数翻译: 旧参数 std/pro → 新版 resolution 值
        _res_map = {"std": "720p", "pro": "1080p", "standard": "720p"}
        mode_in = kwargs.get("mode", "pro")
        resolution = kwargs.get("resolution") or _res_map.get(mode_in, mode_in)
        audio = kwargs.get("sound", "off")      # "on" or "off"
        negative_prompt = kwargs.get("negative_prompt", "")
        aspect_ratio = kwargs.get("aspect_ratio", "16:9")
        cfg_scale = kwargs.get("cfg_scale")

        start_time = time.time()
        is_i2v = bool(img_url or img_path)
        base_url = get_provider_base_url("KLING")

        # 组装 contents (新版)
        contents: list = [{"type": "prompt", "text": prompt}]
        if negative_prompt:
            contents.append({"type": "negative_prompt", "text": negative_prompt})
        if is_i2v:
            image_url = self._resolve_image_input(img_url, img_path)
            contents.append({"type": "first_frame", "url": image_url})

        # 组装 settings (新版)
        settings: Dict[str, Any] = {
            "resolution": resolution,
            "duration": duration,
            "audio": audio,
            "multi_shot": False,
        }
        if cfg_scale is not None:
            settings["cfg_scale"] = cfg_scale
        if aspect_ratio:
            settings["aspect_ratio"] = aspect_ratio

        # options (新版, 可选)
        options: Dict[str, Any] = {}
        if kwargs.get("callback_url"):
            options["callback_url"] = kwargs["callback_url"]

        body = {
            "contents": contents,
            "settings": settings,
            "options": options,
        }

        # 提交任务: POST /image-to-video/{model_id}
        # 1302 限频 / 1303 超并发 / 5xxx 服务端 → 指数退避后重试（≤3 次）
        submit_url = f"{base_url}/image-to-video/{self.model_name}"
        logger.info(f"[Kling] Submitting {'i2v' if is_i2v else 't2v'} task (model={self.model_name})")

        task_id = None
        max_submit_attempts = 3
        for attempt in range(1, max_submit_attempts + 1):
            try:
                response = requests.post(submit_url, headers=self._auth_headers(), json=body, timeout=30)
            except requests.RequestException as e:
                raise _wrap_request_exception(e, "submit") from e
            if response.status_code != 200:
                err = _kling_error_from_response(response, "submit")
                if err.retryable and attempt < max_submit_attempts:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "[Kling] submit retryable error (code=%s http=%s), backoff %.1fs (attempt %d/%d)",
                        err.code, err.status_code, delay, attempt, max_submit_attempts,
                    )
                    time.sleep(delay)
                    continue
                raise err
            task_data = response.json()
            if task_data.get("code") != 0:
                category, retryable = classify_kling_error(code=task_data.get("code"))
                err = KlingError(
                    f"Kling API error (code {task_data.get('code')}): {task_data.get('message', 'unknown error')}",
                    code=task_data.get("code"),
                    message_en=task_data.get("message"),
                    request_id=task_data.get("request_id"),
                    status_code=response.status_code,
                    retryable=retryable,
                    category=category,
                )
                if err.retryable and attempt < max_submit_attempts:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "[Kling] submit retryable error (code=%s), backoff %.1fs (attempt %d/%d)",
                        err.code, delay, attempt, max_submit_attempts,
                    )
                    time.sleep(delay)
                    continue
                raise err
            task_id = task_data["data"]["id"]
            break

        if task_id is None:
            # 理论不可达（循环内要么返回要么 raise）
            raise KlingError("Kling submit failed without response", category="unknown")

        logger.info(f"[Kling] Task submitted: {task_id}")

        # 轮询: GET /tasks?task_ids=xxx
        # 退避节奏：首查 2s，之后 delay=min(delay*1.5, 10)+jitter；
        # 查询类官方码 1302/1303/5xxx 退避后继续轮询（不计入终态）。
        max_wait = 600
        poll_delay = 2.0
        elapsed = 0.0
        poll_url = f"{base_url}/tasks?task_ids={task_id}"

        while elapsed < max_wait:
            time.sleep(poll_delay)
            elapsed += poll_delay

            try:
                resp = requests.get(poll_url, headers=self._auth_headers(), timeout=30)
            except requests.RequestException as e:
                # 单次轮询网络抖动不终止任务——退避后继续
                delay = _backoff_delay(2)
                logger.warning("[Kling] poll network error (%s), backoff %.1fs", e, delay)
                poll_delay = min(poll_delay * 1.5, 10.0)
                elapsed += delay
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                err = _kling_error_from_response(resp, "poll")
                if err.retryable:
                    delay = _backoff_delay(2)
                    logger.warning(
                        "[Kling] poll retryable HTTP %s (code=%s), backoff %.1fs",
                        err.status_code, err.code, delay,
                    )
                    poll_delay = min(poll_delay * 1.5, 10.0)
                    elapsed += delay
                    time.sleep(delay)
                    continue
                raise err
            result_data = resp.json()

            if result_data.get("code") != 0:
                category, retryable = classify_kling_error(code=result_data.get("code"))
                if retryable:
                    delay = _backoff_delay(2)
                    logger.warning(
                        "[Kling] poll retryable error (code=%s), backoff %.1fs",
                        result_data.get("code"), delay,
                    )
                    poll_delay = min(poll_delay * 1.5, 10.0)
                    elapsed += delay
                    time.sleep(delay)
                    continue
                raise KlingError(
                    f"Kling poll error: {result_data.get('message')}",
                    code=result_data.get("code"),
                    message_en=result_data.get("message"),
                    request_id=result_data.get("request_id"),
                    status_code=resp.status_code,
                    retryable=retryable,
                    category=category,
                )

            tasks = result_data.get("data", [])
            if not tasks:
                logger.warning(f"[Kling] No task data yet ({elapsed:.0f}s)")
                poll_delay = min(poll_delay * 1.5, 10.0)
                continue

            status = tasks[0].get("status")
            logger.info(f"[Kling] Task status: {status} ({elapsed:.0f}s)")

            if status == "succeeded":
                outputs = tasks[0].get("outputs", [])
                video_url = None
                for out in outputs:
                    if out.get("type") == "video" and out.get("url"):
                        video_url = out["url"]
                        break
                if not video_url and outputs:
                    video_url = outputs[0].get("url")

                if not video_url:
                    raise KlingError(
                        "Kling task succeeded but no video URL found",
                        retryable=False,
                        category="unknown",
                    )

                # 下载视频（临时 URL，需立即下载）
                try:
                    video_content = requests.get(video_url, timeout=120).content
                except requests.RequestException as e:
                    raise _wrap_request_exception(e, "video download") from e
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(video_content)

                generation_time = time.time() - start_time
                logger.info(f"[Kling] Done in {generation_time:.1f}s -> {output_path}")
                return output_path, generation_time

            elif status in _TERMINAL_TASK_STATUSES:
                task_code = tasks[0].get("code")
                category, retryable = classify_kling_error(code=task_code)
                raise KlingError(
                    f"Kling task {status}: {tasks[0].get('message', 'Unknown error')}",
                    code=task_code,
                    message_en=tasks[0].get("message"),
                    request_id=tasks[0].get("request_id") or tasks[0].get("task_id"),
                    retryable=retryable,
                    category=category if status == "failed" else "terminal",
                )

            poll_delay = min(poll_delay * 1.5, 10.0)

        raise KlingError(
            f"Kling task timed out after {max_wait}s",
            retryable=False,
            category="timeout",
        )
