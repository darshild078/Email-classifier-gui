import tkinter as tk
from tkinter import ttk

class SpamDetectorTheme:
    """Central theme configuration for spam detector GUI."""

    # Color palette
    COLORS = {
        'primary': '#2E86AB', 'primary_dark': '#1B4D72', 'secondary': '#A23B72',
        'success': '#28A745', 'danger': '#DC3545', 'warning': '#FFC107', 'info': '#17A2B8',
        'light_gray': '#F8F9FA', 'medium_gray': '#6C757D', 'dark_gray': '#343A40',
        'border': '#DEE2E6', 'hover': '#E3F2FD', 'active': '#BBDEFB', 'disabled': '#E9ECEF'
    }

    # Fonts
    FONTS = {
        'title': ('Segoe UI', 18, 'bold'),
        'heading': ('Segoe UI', 14, 'bold'),
        'body': ('Segoe UI', 10, 'normal'),
        'body_bold': ('Segoe UI', 10, 'bold'),
        'monospace': ('Consolas', 9, 'normal'),
        'small': ('Segoe UI', 8, 'normal')
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

    # Primary buttons
    style.configure('Modern.TButton',
                    background=theme.COLORS['primary'],
                    foreground='white',
                    borderwidth=0,
                    font=theme.FONTS['body_bold'])
    style.map('Modern.TButton',
              background=[('active', theme.COLORS['primary_dark']),
                          ('pressed', theme.COLORS['primary_dark'])])

    # Danger buttons (e.g., clear/reset)
    style.configure('Danger.TButton',
                    background=theme.COLORS['danger'],
                    foreground='white',
                    borderwidth=0,
                    font=theme.FONTS['body'])
    style.map('Danger.TButton',
              background=[('active', '#C82333'), ('pressed', '#C82333')])

    # Success buttons
    style.configure('Success.TButton',
                    background=theme.COLORS['success'],
                    foreground='white',
                    borderwidth=0,
                    font=theme.FONTS['body_bold'])

    # Label frames
    style.configure('Modern.TLabelframe',
                    background=theme.COLORS['light_gray'],
                    borderwidth=1, relief='solid')
    style.configure('Modern.TLabelframe.Label',
                    background=theme.COLORS['light_gray'],
                    foreground=theme.COLORS['dark_gray'],
                    font=theme.FONTS['heading'])

    return style, theme

def create_status_indicator(parent, text="Ready", status="info"):
    """Create a status widget with colored dot and text."""
    theme = SpamDetectorTheme()
    frame = tk.Frame(parent, background=theme.COLORS['light_gray'])

    status_colors = {
        'info': theme.COLORS['info'],
        'success': theme.COLORS['success'],
        'warning': theme.COLORS['warning'],
        'danger': theme.COLORS['danger']
    }

    # Colored dot
    canvas = tk.Canvas(frame, width=12, height=12,
                       background=theme.COLORS['light_gray'],
                       highlightthickness=0)
    canvas.create_oval(2, 2, 10, 10, fill=status_colors.get(status, theme.COLORS['info']), outline='')
    canvas.pack(side=tk.LEFT, padx=(0, theme.SPACING['sm']))

    # Status text
    label = tk.Label(frame, text=text,
                     background=theme.COLORS['light_gray'],
                     foreground=theme.COLORS['dark_gray'],
                     font=theme.FONTS['body'])
    label.pack(side=tk.LEFT)

    return frame, label

class ModernScrolledText(tk.Frame):
    """Custom scrolled text widget with professional theme integration."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        theme = SpamDetectorTheme()

        self.text_widget = tk.Text(
            self,
            wrap=kwargs.get('wrap', tk.WORD),
            height=kwargs.get('height', 10),
            width=kwargs.get('width', 50),
            font=kwargs.get('font', theme.FONTS['body']),
            background=kwargs.get('background', 'white'),
            foreground=kwargs.get('foreground', theme.COLORS['dark_gray']),
            borderwidth=1,
            relief='solid',
            selectbackground=theme.COLORS['active']
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
