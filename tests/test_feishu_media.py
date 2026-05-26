from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from smzdm_notice.feishu import media as feishu_media


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

    def test_download_image_rejects_non_image_or_oversized_response(self) -> None:
        response = Mock()
        response.headers = {"content-type": "text/html"}
        response.content = b"<html></html>"
        response.raise_for_status.return_value = None
        client = Mock()
        client.get.return_value = response
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=None)

        with patch("smzdm_notice.feishu.media.httpx.Client", return_value=client), self.assertRaises(ValueError):
            feishu_media.download_image("https://img.example.com/a.jpg")

        response.headers = {"content-type": "image/jpeg"}
        response.content = b"x" * (feishu_media.IMAGE_MAX_BYTES + 1)
        with patch("smzdm_notice.feishu.media.httpx.Client", return_value=client), self.assertRaises(ValueError):
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
