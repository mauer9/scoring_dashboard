def get_sidebar_style():
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 75.0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "height": "100%",
        "z-index": 1,
        "overflow-x": "hidden",
        "transition": "all 0.5s",
        "padding": "0.5rem 1rem",
        "background-color": "#111111",
    }
    return SIDEBAR_STYLE

def get_sidebar_hidden_style():
    SIDEBAR_HIDDEN = {
        "position": "fixed",
        "top": 62.5,
        "left": "-16rem",
        "bottom": 0,
        "width": "16rem",
        "height": "100%",
        "z-index": 1,
        "overflow-x": "hidden",
        "transition": "all 0.5s",
        "padding": "0rem 0rem",
        "background-color": "#111111",
    }
    return SIDEBAR_HIDDEN

def get_content_style():
    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "transition": "margin-left .5s",
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "background-color": "#111111",
    }

    CONTENT_STYLE1 = {
        "transition": "margin-left .5s",
        "margin-left": "2rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "background-color": "#111111",
    }
    return CONTENT_STYLE, CONTENT_STYLE1

def get_footer_style():
    FOOTER_STYLE = {
        "position": "absolute",
        "bottom": "0",
        "width": "100%",
        "height": "60px",  # /* Set the fixed height of the footer here */
        "line-height": "60px",  # /* Vertically center the text there */
        "background-color": "#111111"
    }
    return FOOTER_STYLE

def get_modal_style():
    MODAL = {
        "position": "fixed",
        "z-index": "1002",  # /* Sit on top, including modebar which has z=1001 */
        "left": "0",
        "top": "0",
        "width": "100%",  # /* Full width */
        "height": "100%",  # /* Full height */
        "background-color": "rgba(0, 0, 0, 0.6)",
        "display": "block"  # /* Black w/ opacity */
    }
    return MODAL