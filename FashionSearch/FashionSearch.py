"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx

from rxconfig import config

from .ui.pages.home_page import home_page


class State(rx.State):
    """The app state."""

    ...


app = rx.App()
