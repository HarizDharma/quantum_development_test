# Test just the bracket/parenthesis structure
def html_div(children=None, style=None):
    return f"Div({children}, {style})"

def dcc_store(id=None):
    return f"Store({id})"

# Test the problematic structure - this should match the pattern from app_dash.py
layout = html_div([
    # Global stores
    dcc_store(id="data-store"),
    
    # Main layout container  
    html_div([
        # Header
        html_div("header"),
        # Content  
        html_div([
            html_div("content")
        ], style={
            "padding": "20px"
        })
    ]),
    
    # Hidden elements
    html_div([
        html_div(style={"display": "none"})
    ])
], style={
    "fontFamily": "Inter"
})

print("Syntax test passed!")
print(f"Layout: {layout}")