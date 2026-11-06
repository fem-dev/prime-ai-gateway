from pathlib import Path

import anywidget
import traitlets


class PrimeChatWidget(anywidget.AnyWidget):
    """Custom Prime AI chat surface; independent of mo.ui.chat."""

    _esm = Path(__file__).with_name("prime_chat.js")
    _css = Path(__file__).with_name("prime_chat.css")

    request = traitlets.Unicode("").tag(sync=True)
    cancel_request = traitlets.Unicode("").tag(sync=True)
    response = traitlets.Unicode("").tag(sync=True)
    status = traitlets.Unicode("آماده").tag(sync=True)
