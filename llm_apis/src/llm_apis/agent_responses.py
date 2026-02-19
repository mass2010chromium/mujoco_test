import base64

from typing import List

import cv2

class StringResponse:
    
    def __init__(self, s: str, role: str = 'assistant'):
        self.msg: str = s
        self.role: str = role

    @staticmethod
    def tool_response(string: str) -> str:
        return f'```tool_output\n{string}\n```'''

    def to_msg(self, **kwargs) -> dict:
        return {'role': self.role, 'content': StringResponse.tool_response(self.msg)}

    def unbox(self):
        return self.msg

class IntResponse(StringResponse):
    def __init__(self, i: int, role: str = 'assistant'):
        super().__init__(str(i), role)
        self.val = i

    def unbox(self):
        return self.val


class ImageResponse(StringResponse):
    def __init__(self, img: 'rgb_image' = None, images: List['rgb_image'] = None, message: str = None, role: str = 'assistant'):
        """
        Expects RGB array
        """
        if img is not None:
            assert images is None
        # On this branch, img is None.
        elif images is None:
            raise ValueError("One of img or images must not be None")
        self.img = img
        self.images = images
        self.role = role
        self.message = message

    @staticmethod
    def encode_image(image_rgb):
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        res, buf = cv2.imencode(".png", image_bgr)
        return base64.b64encode(buf).decode('utf-8')

    def to_msg(self, **kwargs) -> dict:
        message = self.message
        if message is None:
            message = 'See attached images.'
        image_dat = self.images if self.images is not None else [self.img]
        if kwargs.get('raw', False):
            image_resp = image_dat
        else:
            image_resp = [ImageResponse.encode_image(img) for img in image_dat]
        return {
            'role': self.role,
            'content': StringResponse.tool_response(message),
            'images': image_resp
        }

    def unbox(self):
        if self.img is not None:
            return self.img
        return self.images


