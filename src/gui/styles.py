import tkinter as tk
from tkinter import ttk

class SpamDetectorTheme:
    """Central theme configuration for spam detector GUI."""

    # Codex-inspired professional palette: neutral surfaces, warm text, quiet accent.
    COLORS = {
        'background': '#0F1115',
        'surface': '#161A20',
        'surface_raised': '#1E232B',
        'surface_muted': '#12151A',
        'primary': '#7DD3C7',
        'primary_dark': '#46A89A',
        'primary_soft': '#193D3A',
        'secondary': '#D8C7A1',
        'success': '#7BC99E',
        'danger': '#EF7D7D',
        'danger_dark': '#D86565',
        'warning': '#E8C76A',
        'info': '#85B8FF',
        'text': '#F4F1E8',
        'text_muted': '#A7ADB7',
        'text_subtle': '#767D89',
        'border': '#2A303A',
        'border_strong': '#3A414E',
        'hover': '#252B34',
        'active': '#2F5D59',
        'disabled': '#272C34',
        # Backward-compatible aliases used throughout older widgets.
        'light_gray': '#0F1115',
        'medium_gray': '#A7ADB7',
        'dark_gray': '#F4F1E8',
    }

    # Fonts
    FONTS = {
        'title': ('Segoe UI Semibold', 22, 'normal'),
        'heading': ('Segoe UI Semibold', 13, 'normal'),
        'body': ('Segoe UI', 10, 'normal'),
        'body_bold': ('Segoe UI Semibold', 10, 'normal'),
        'monospace': ('Cascadia Mono', 9, 'normal'),
        'small': ('Segoe UI', 9, 'normal')
    }

    # Spacing values
    SPACING = {'xs': 4, 'sm': 8, 'md': 12, 'lg': 16, 'xl': 24, 'xxl': 32}

    # Component sizes
    SIZES = {
        'button_height': 32, 'input_height': 28,
        'window_min_width': 700, 'window_min_height': 500,
        'text_area_height': 150
    }

def apply_modern_style():
    """Apply modern styles to ttk widgets for professional look."""
    style = ttk.Style()
    
    try:
        style.theme_use('clam')  # Modern base theme
    except:
        style.theme_use('default')
    
    theme = SpamDetectorTheme()

    style.configure('.',
                    background=theme.COLORS['background'],
                    foreground=theme.COLORS['text'],
                    font=theme.FONTS['body'])
    style.configure('TFrame', background=theme.COLORS['background'])
    style.configure('Surface.TFrame', background=theme.COLORS['surface'])
    style.configure('TLabel',
                    background=theme.COLORS['background'],
                    foreground=theme.COLORS['text'],
                    font=theme.FONTS['body'])

    # Primary buttons
    style.configure('Modern.TButton',
                    background=theme.COLORS['primary'],
                    foreground=theme.COLORS['background'],
                    borderwidth=0,
                    focusthickness=0,
                    padding=(16, 9),
                    font=theme.FONTS['body_bold'])
    style.map('Modern.TButton',
              background=[('active', theme.COLORS['primary_dark']),
                          ('pressed', theme.COLORS['primary_dark']),
                          ('disabled', theme.COLORS['disabled'])],
              foreground=[('disabled', theme.COLORS['text_subtle'])])

    style.configure('Secondary.TButton',
                    background=theme.COLORS['surface_raised'],
                    foreground=theme.COLORS['text'],
                    borderwidth=1,
                    bordercolor=theme.COLORS['border'],
                    focusthickness=0,
                    padding=(14, 8),
                    font=theme.FONTS['body'])
    style.map('Secondary.TButton',
              background=[('active', theme.COLORS['hover']),
                          ('pressed', theme.COLORS['hover']),
                          ('disabled', theme.COLORS['disabled'])],
              foreground=[('disabled', theme.COLORS['text_subtle'])])

    # Danger buttons (e.g., clear/reset)
    style.configure('Danger.TButton',
                    background=theme.COLORS['danger'],
                    foreground=theme.COLORS['background'],
                    borderwidth=0,
                    focusthickness=0,
                    padding=(14, 8),
                    font=theme.FONTS['body'])
    style.map('Danger.TButton',
              background=[('active', theme.COLORS['danger_dark']),
                          ('pressed', theme.COLORS['danger_dark'])])

    # Success buttons
    style.configure('Success.TButton',
                    background=theme.COLORS['success'],
                    foreground=theme.COLORS['background'],
                    borderwidth=0,
                    padding=(14, 8),
                    font=theme.FONTS['body_bold'])

    # Label frames
    style.configure('Modern.TLabelframe',
                    background=theme.COLORS['surface'],
                    bordercolor=theme.COLORS['border'],
                    borderwidth=1,
                    relief='solid')
    style.configure('Modern.TLabelframe.Label',
                    background=theme.COLORS['surface'],
                    foreground=theme.COLORS['text'],
                    font=theme.FONTS['heading'])
    style.configure('TPanedwindow', background=theme.COLORS['background'])
    style.configure('Sash', background=theme.COLORS['border'])
    style.configure('Vertical.TScrollbar',
                    background=theme.COLORS['surface_raised'],
                    troughcolor=theme.COLORS['surface_muted'],
                    bordercolor=theme.COLORS['border'],
                    arrowcolor=theme.COLORS['text_muted'])

    return style, theme

def create_status_indicator(parent, text="Ready", status="info", background=None):
    """Create a status widget with colored dot and text."""
    theme = SpamDetectorTheme()
    bg = background or theme.COLORS['background']
    frame = tk.Frame(parent, background=bg)

    status_colors = {
        'info': theme.COLORS['info'],
        'success': theme.COLORS['success'],
        'warning': theme.COLORS['warning'],
        'danger': theme.COLORS['danger']
    }

    # Colored dot
    canvas = tk.Canvas(frame, width=12, height=12,
                       background=bg,
                       highlightthickness=0)
    canvas.create_oval(2, 2, 10, 10, fill=status_colors.get(status, theme.COLORS['info']), outline='')
    canvas.pack(side=tk.LEFT, padx=(0, theme.SPACING['sm']))

    # Status text
    label = tk.Label(frame, text=text,
                     background=bg,
                     foreground=theme.COLORS['text_muted'],
                     font=theme.FONTS['body'])
    label.pack(side=tk.LEFT)

    return frame, label

class ModernScrolledText(tk.Frame):
    """Custom scrolled text widget with professional theme integration."""

    def __init__(self, parent, **kwargs):
        theme = SpamDetectorTheme()
        super().__init__(parent, background=kwargs.get('frame_background', theme.COLORS['border']))

        self.text_widget = tk.Text(
            self,
            wrap=kwargs.get('wrap', tk.WORD),
            height=kwargs.get('height', 10),
            width=kwargs.get('width', 50),
            font=kwargs.get('font', theme.FONTS['body']),
            background=kwargs.get('background', theme.COLORS['surface_muted']),
            foreground=kwargs.get('foreground', theme.COLORS['text']),
            insertbackground=theme.COLORS['text'],
            borderwidth=0,
            relief='flat',
            padx=14,
            pady=12,
            selectbackground=theme.COLORS['active'],
            selectforeground=theme.COLORS['text']
        )

        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    # Basic text operations
    def get(self, start, end=None): return self.text_widget.get(start, end)
    def insert(self, index, text, *tags): return self.text_widget.insert(index, text, *tags)
    def delete(self, start, end=None): return self.text_widget.delete(start, end)

    def configure_tags(self):
        """Set up tags for colored outputs (spam, ham, info, warning)."""
        theme = SpamDetectorTheme()
        self.text_widget.tag_configure("spam", foreground=theme.COLORS['danger'], font=theme.FONTS['body_bold'])
        self.text_widget.tag_configure("ham", foreground=theme.COLORS['success'], font=theme.FONTS['body_bold'])
        self.text_widget.tag_configure("info", foreground=theme.COLORS['info'])
        self.text_widget.tag_configure("warning", foreground=theme.COLORS['warning'])
        self.text_widget.tag_configure("monospace", font=theme.FONTS['monospace'])
