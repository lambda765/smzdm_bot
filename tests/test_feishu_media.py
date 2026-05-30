from __future__ import annotations

import unittest
from collections import deque
from unittest.mock import Mock, patch

from smzdm_notice.feishu import media as feishu_media


class _StreamResponse:
    def __init__(
        self,
        url: str,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = headers or {"content-type": "image/jpeg"}
        self._chunks = chunks or [b"image-bytes"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        yield from self._chunks


class _StreamClient:
    def __init__(self, responses: list[_StreamResponse]) -> None:
        self.responses = deque(responses)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def stream(self, method: str, url: str):
        response = self.responses.popleft()
        response.url = url
        return response


class FeishuMediaTests(unittest.TestCase):
    def setUp(self) -> None:
        feishu_media.IMAGE_KEY_CACHE.clear()

    def test_image_key_cache_avoids_reuploading_same_url(self) -> None:
        with (
            patch("smzdm_notice.feishu.media.download_image", return_value=b"image-bytes") as download_image,
            patch("smzdm_notice.feishu.media.upload_image", return_value="img_cached") as upload_image,
        ):
            self.assertEqual(feishu_media.get_feishu_image_key("https://img.example.com/a.jpg"), "img_cached")
            self.assertEqual(feishu_media.get_feishu_image_key("https://img.example.com/a.jpg"), "img_cached")

        download_image.assert_called_once_with("https://img.example.com/a.jpg")
        upload_image.assert_called_once_with(b"image-bytes")

    def test_image_key_cache_evicts_oldest_entry_when_full(self) -> None:
        for index in range(1001):
            feishu_media.IMAGE_KEY_CACHE[f"https://img.example.com/{index}.jpg"] = f"img_{index}"

        self.assertEqual(len(feishu_media.IMAGE_KEY_CACHE), 1000)
        self.assertNotIn("https://img.example.com/0.jpg", feishu_media.IMAGE_KEY_CACHE)
        self.assertEqual(feishu_media.IMAGE_KEY_CACHE["https://img.example.com/1000.jpg"], "img_1000")

    def test_download_image_rejects_non_image_or_oversized_response(self) -> None:
        non_image_client = _StreamClient(
            [_StreamResponse("https://img.example.com/a.jpg", headers={"content-type": "text/html"})]
        )
        oversized_client = _StreamClient(
            [
                _StreamResponse(
                    "https://img.example.com/a.jpg",
                    chunks=[b"x" * (feishu_media.IMAGE_MAX_BYTES + 1)],
                )
            ]
        )

        with (
            patch(
                "smzdm_notice.feishu.media.socket.getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 0))]
            ),
            patch("smzdm_notice.feishu.media.httpx.Client", return_value=non_image_client),
            self.assertRaises(ValueError),
        ):
            feishu_media.download_image("https://img.example.com/a.jpg")

        with (
            patch(
                "smzdm_notice.feishu.media.socket.getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 0))]
            ),
            patch("smzdm_notice.feishu.media.httpx.Client", return_value=oversized_client),
            self.assertRaises(ValueError),
        ):
            feishu_media.download_image("https://img.example.com/a.jpg")

    def test_download_image_rejects_non_public_hosts(self) -> None:
        blocked_urls = [
            "http://localhost/a.jpg",
            "http://127.0.0.1/a.jpg",
            "http://[::1]/a.jpg",
            "http://10.0.0.1/a.jpg",
        ]

        for url in blocked_urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                feishu_media.download_image(url)

    def test_download_image_rejects_domain_resolving_to_private_ip(self) -> None:
        with (
            patch(
                "smzdm_notice.feishu.media.socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 0))]
            ),
            self.assertRaises(ValueError),
        ):
            feishu_media.download_image("https://img.example.com/a.jpg")

    def test_download_image_validates_redirect_targets(self) -> None:
        client = _StreamClient(
            [
                _StreamResponse(
                    "https://img.example.com/a.jpg", status_code=302, headers={"location": "http://127.0.0.1/a.jpg"}
                ),
            ]
        )

        with (
            patch(
                "smzdm_notice.feishu.media.socket.getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 0))]
            ),
            patch("smzdm_notice.feishu.media.httpx.Client", return_value=client),
            self.assertRaises(ValueError),
        ):
            feishu_media.download_image("https://img.example.com/a.jpg")

    def test_download_image_supports_relative_redirects(self) -> None:
        client = _StreamClient(
            [
                _StreamResponse("https://img.example.com/a.jpg", status_code=302, headers={"location": "/b.jpg"}),
                _StreamResponse("https://img.example.com/b.jpg", chunks=[b"ok"]),
            ]
        )

        with (
            patch(
                "smzdm_notice.feishu.media.socket.getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 0))]
            ),
            patch("smzdm_notice.feishu.media.httpx.Client", return_value=client),
        ):
            self.assertEqual(feishu_media.download_image("https://img.example.com/a.jpg"), b"ok")

    def test_download_image_rejects_too_many_redirects(self) -> None:
        client = _StreamClient(
            [
                _StreamResponse("https://img.example.com/a.jpg", status_code=302, headers={"location": "/b.jpg"}),
                _StreamResponse("https://img.example.com/b.jpg", status_code=302, headers={"location": "/c.jpg"}),
                _StreamResponse("https://img.example.com/c.jpg", status_code=302, headers={"location": "/d.jpg"}),
                _StreamResponse("https://img.example.com/d.jpg", status_code=302, headers={"location": "/e.jpg"}),
                _StreamResponse("https://img.example.com/e.jpg", status_code=302, headers={"location": "/f.jpg"}),
                _StreamResponse("https://img.example.com/f.jpg", status_code=302, headers={"location": "/g.jpg"}),
            ]
        )

        with (
            patch(
                "smzdm_notice.feishu.media.socket.getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 0))]
            ),
            patch("smzdm_notice.feishu.media.httpx.Client", return_value=client),
            self.assertRaises(ValueError),
        ):
            feishu_media.download_image("https://img.example.com/a.jpg")

    def test_upload_image_returns_image_key(self) -> None:
        CreateImageRequest = Mock()
        request_builder = Mock()
        request_builder.request_body.return_value = request_builder
        request_builder.build.return_value = "request"
        CreateImageRequest.builder.return_value = request_builder

        CreateImageRequestBody = Mock()
        body_builder = Mock()
        body_builder.image_type.return_value = body_builder
        body_builder.image.return_value = body_builder
        body_builder.build.return_value = "body"
        CreateImageRequestBody.builder.return_value = body_builder

        response = Mock()
        response.success.return_value = True
        response.data.image_key = "img_key"
        client = Mock()
        client.im.v1.image.create.return_value = response

        with (
            patch(
                "smzdm_notice.feishu.media.get_image_models",
                return_value=(CreateImageRequest, CreateImageRequestBody),
            ),
            patch("smzdm_notice.feishu.media.get_lark_client", return_value=client),
        ):
            self.assertEqual(feishu_media.upload_image(b"image-bytes"), "img_key")

        body_builder.image_type.assert_called_once_with("message")
        body_builder.image.assert_called_once()
        client.im.v1.image.create.assert_called_once_with("request")


if __name__ == "__main__":
    unittest.main()
