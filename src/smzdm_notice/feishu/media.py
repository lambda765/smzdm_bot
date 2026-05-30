"""飞书媒体资源处理。"""

from __future__ import annotations

import ipaddress
import socket
from collections import OrderedDict
from io import BytesIO
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

from smzdm_notice.feishu.sdk import get_image_models, get_lark_client

_IMAGE_KEY_CACHE_MAX = 1000


class _ImageKeyCache(OrderedDict[str, str]):
    def __setitem__(self, key: str, value: str) -> None:
        super().__setitem__(key, value)
        if len(self) > _IMAGE_KEY_CACHE_MAX:
            self.popitem(last=False)


IMAGE_KEY_CACHE: _ImageKeyCache = _ImageKeyCache()
IMAGE_MAX_BYTES = 10 * 1024 * 1024
IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 5


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
    url = _normalize_and_validate_image_url(image_url)
    with httpx.Client(timeout=10.0, follow_redirects=False) as client:
        redirect_count = 0
        while True:
            with client.stream("GET", url) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    if redirect_count >= _MAX_REDIRECTS:
                        raise ValueError("重定向次数过多")
                    location = response.headers.get("location", "")
                    if not location:
                        raise ValueError("重定向缺少 Location")
                    url = _normalize_and_validate_image_url(location, base_url=str(response.url))
                    redirect_count += 1
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower().strip()
                if content_type not in IMAGE_CONTENT_TYPES:
                    raise ValueError(f"不支持的图片类型: {content_type or 'unknown'}")
                content = _read_limited_response(response)
                break
    if not content:
        raise ValueError("图片内容为空")
    return content


def _normalize_and_validate_image_url(url: str, base_url: str | None = None) -> str:
    normalized = str(url or "").strip()
    if base_url and not urlparse(normalized).scheme:
        normalized = urljoin(base_url, normalized)
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"不允许的 URL 协议: {parsed.scheme or 'unknown'}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL 缺少主机名")
    hostname = hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("不允许访问 localhost")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        _validate_resolved_addresses(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    else:
        _validate_global_ip(ip, hostname)
    return normalized


def _validate_resolved_addresses(hostname: str, port: int) -> None:
    try:
        addrinfos = socket.getaddrinfo(hostname, port)
    except socket.gaierror as e:
        raise ValueError(f"无法解析图片域名: {hostname}") from e
    for _family, _socktype, _proto, _canonname, sockaddr in addrinfos:
        _validate_global_ip(ipaddress.ip_address(sockaddr[0]), hostname)


def _validate_global_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, label: str) -> None:
    if not ip.is_global:
        raise ValueError(f"不允许访问非公网地址: {label}")


def _read_limited_response(response: httpx.Response) -> bytes:
    chunks = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > IMAGE_MAX_BYTES:
            raise ValueError(f"图片超过 {IMAGE_MAX_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


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
