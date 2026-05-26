"""飞书媒体资源处理。"""

from __future__ import annotations

from io import BytesIO

import httpx
from loguru import logger

from smzdm_notice.feishu.sdk import get_image_models, get_lark_client

IMAGE_KEY_CACHE: dict[str, str] = {}
IMAGE_MAX_BYTES = 10 * 1024 * 1024
IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}


def get_feishu_image_key(image_url: str) -> str:
    """下载远程图片并上传飞书，返回可用于卡片展示的 image_key。"""
    url = str(image_url or "").strip()
    if not url:
        return ""
    cached = IMAGE_KEY_CACHE.get(url)
    if cached:
        return cached
    try:
        image_bytes = download_image(url)
        image_key = upload_image(image_bytes)
    except Exception as e:
        logger.warning(f"商品图片处理失败，将保留图片链接: {e}")
        return ""
    if image_key:
        IMAGE_KEY_CACHE[url] = image_key
    return image_key


def download_image(image_url: str) -> bytes:
    """下载图片并限制类型和大小，避免上传非图片或过大内容。"""
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        response = client.get(image_url)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower().strip()
    if content_type not in IMAGE_CONTENT_TYPES:
        raise ValueError(f"不支持的图片类型: {content_type or 'unknown'}")
    content = response.content
    if not content:
        raise ValueError("图片内容为空")
    if len(content) > IMAGE_MAX_BYTES:
        raise ValueError(f"图片超过 10MB: {len(content)} bytes")
    return content


def upload_image(image_bytes: bytes) -> str:
    """上传图片到飞书并返回 image_key。"""
    CreateImageRequest, CreateImageRequestBody = get_image_models()
    request = (
        CreateImageRequest.builder()
        .request_body(CreateImageRequestBody.builder().image_type("message").image(BytesIO(image_bytes)).build())
        .build()
    )
    response = get_lark_client().im.v1.image.create(request)
    if response.success():
        image_key = str(getattr(response.data, "image_key", "") or "")
        if image_key:
            return image_key
        raise ValueError("飞书图片上传成功但未返回 image_key")
    raise RuntimeError(f"飞书图片上传失败: code={response.code}, msg={response.msg}")
