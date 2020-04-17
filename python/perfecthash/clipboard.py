#===============================================================================
# Imports
#===============================================================================
import os
import sys

from .util import (
    is_win32,
    unicode_class,
)

#===============================================================================
# Main
#===============================================================================

if is_win32:
    from win32con import (
        CF_UNICODETEXT,
    )

    from win32clipboard import (
        OpenClipboard,
        CloseClipboard,
        EmptyClipboard,
        GetClipboardData,
        SetClipboardData,
    )

    def copy(fmt=CF_UNICODETEXT):
        OpenClipboard()
        text = GetClipboardData(fmt)
        CloseClipboard()
        return text

    def paste(text, fmt=CF_UNICODETEXT, cls=unicode_class):
        EmptyClipboard()
        data = cls(text) if cls and not isinstance(text, cls) else text
        SetClipboardData(fmt, data)
        CloseClipboard()
        return data

    def cb(text=None, fmt=CF_UNICODETEXT, cls=unicode_class):
        OpenClipboard()
        if not text:
            text = GetClipboardData(fmt)
            CloseClipboard()
            return text
        EmptyClipboard()
        data = cls(text) if cls and not isinstance(text, cls) else text
        SetClipboardData(fmt, data)
        CloseClipboard()
        return data


# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
