import base64
from functools import wraps

from typing import List

import cv2

class StringResponse:
    
    def __init__(self, s: str, role: str = 'assistant'):
        self.msg: str = s
        self.role: str = role

    @staticmethod
    def tool_response(string: str) -> str:
        return f'```tool_output\n{string}\n```'''

    def to_msg(self) -> dict:
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
        if img is None:
            assert images is not None
        elif images is None:
            assert img is not None
        else:
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

    def to_msg(self) -> dict:
        message = self.message
        if self.images is not None:
            if message is None:
                message = 'See attached images.'
            return {
                'role': self.role,
                'content': StringResponse.tool_response(message),
                'images': [ImageResponse.encode_image(img) for img in self.images]
            }
        if message is None:
            message = 'See attached image.'
        return {
            'role': self.role,
            'content': StringResponse.tool_response(message),
            'images': [ImageResponse.encode_image(self.img)]
        }

    def unbox(self):
        if self.img is not None:
            return self.img
        return self.images

def tool_wrap(response_type=StringResponse, role: str = 'assistant'):
    """
    Decorator for wrapping tool execution.
    """
    def func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return response_type(f(*args, **kwargs), role=role)
        return wrapper
    return func



