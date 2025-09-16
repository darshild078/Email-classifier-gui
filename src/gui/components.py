import tkinter as tk
from tkinter import ttk
from .styles import SpamDetectorTheme, ModernScrolledText, create_status_indicator

class MessageInputPanel(tk.Frame):
    """Reusable input panel for pasting and analyzing messages."""

    def __init__(self, parent, on_analyze_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.theme = SpamDetectorTheme()
        self.on_analyze_callback = on_analyze_callback
        self.placeholder_text = "Paste your message here to check if it's spam..."
        self._create_widgets()
        self._setup_placeholder()

    def _create_widgets(self):
        """Create widgets: title, text area, buttons, sample messages."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Title label
        title_label = tk.Label(
            self,
            text="📝 Message Input",
            font=self.theme.FONTS['heading'],
            background=self.theme.COLORS['light_gray'],
            foreground=self.theme.COLORS['dark_gray']
        )
        title_label.grid(row=0, column=0, sticky='w', pady=(0, self.theme.SPACING['sm']))

        # Text input area
        self.text_input = ModernScrolledText(self, height=8, wrap=tk.WORD, font=self.theme.FONTS['body'])
        self.text_input.grid(row=1, column=0, sticky='nsew', pady=(0, self.theme.SPACING['md']))
        self.text_input.configure_tags()

        # Buttons frame
        button_frame = tk.Frame(self, background=self.theme.COLORS['light_gray'])
        button_frame.grid(row=2, column=0, sticky='ew')
        button_frame.columnconfigure(1, weight=1)

        # Analyze button
        self.analyze_button = ttk.Button(
            button_frame, text="🔍 Analyze Message", style='Modern.TButton',
            command=self._on_analyze_clicked, state=tk.DISABLED
        )
        self.analyze_button.grid(row=0, column=0, padx=(0, self.theme.SPACING['sm']))

        # Clear button
        self.clear_button = ttk.Button(button_frame, text="🗑️ Clear", style='Danger.TButton', command=self.clear_text)
        self.clear_button.grid(row=0, column=2, sticky='e')

        # Sample buttons
        sample_frame = tk.Frame(button_frame, background=self.theme.COLORS['light_gray'])
        sample_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(self.theme.SPACING['sm'], 0))
        tk.Label(
            sample_frame, text="Try samples:", font=self.theme.FONTS['small'],
            background=self.theme.COLORS['light_gray'], foreground=self.theme.COLORS['medium_gray']
        ).pack(side=tk.LEFT, padx=(0, self.theme.SPACING['sm']))

        ttk.Button(sample_frame, text="📩 Ham Example", command=self.load_ham_sample).pack(side=tk.LEFT, padx=(0, self.theme.SPACING['xs']))
        ttk.Button(sample_frame, text="🚨 Spam Example", command=self.load_spam_sample).pack(side=tk.LEFT)

    def _setup_placeholder(self):
        """Insert placeholder and bind events."""
        self.text_input.insert("1.0", self.placeholder_text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['medium_gray'])
        self.text_input.text_widget.bind("<FocusIn>", self._on_focus_in)
        self.text_input.text_widget.bind("<FocusOut>", self._on_focus_out)
        self.text_input.text_widget.bind("<KeyRelease>", self._on_text_change)

    def _on_focus_in(self, event):
        """Remove placeholder on focus."""
        if self.get_text() == self.placeholder_text:
            self.text_input.delete("1.0", tk.END)
            self.text_input.text_widget.configure(foreground=self.theme.COLORS['dark_gray'])

    def _on_focus_out(self, event):
        """Restore placeholder if empty."""
        if not self.get_text():
            self.text_input.insert("1.0", self.placeholder_text)
            self.text_input.text_widget.configure(foreground=self.theme.COLORS['medium_gray'])

    def _on_text_change(self, event):
        """Enable analyze button if there is text."""
        text = self.get_text()
        has_content = text and text != self.placeholder_text
        self.analyze_button.configure(state=tk.NORMAL if has_content else tk.DISABLED)

    def _on_analyze_clicked(self):
        """Invoke callback on analyze."""
        if self.on_analyze_callback:
            text = self.get_text()
            if text and text != self.placeholder_text:
                self.on_analyze_callback(text)

    # Utility methods
    def get_text(self): return self.text_input.get("1.0", tk.END).strip()
    def set_text(self, text):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['dark_gray'])
        self._on_text_change(None)
    def clear_text(self):
        self.set_text(self.placeholder_text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['medium_gray'])
        self.analyze_button.configure(state=tk.DISABLED)
    def load_ham_sample(self):
        sample = "Hi! Let's meet tomorrow for coffee. Looking forward to catching up!"
        self.set_text(sample)
    def load_spam_sample(self):
        sample = "CONGRATULATIONS! You've WON $5000 CASH PRIZE! Click here NOW!"
        self.set_text(sample)


class ResultsDisplayPanel(tk.Frame):
    """Reusable panel to display spam analysis results."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.theme = SpamDetectorTheme()
        self._create_widgets()

    def _create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        title_label = tk.Label(
            self, text="📊 Analysis Results", font=self.theme.FONTS['heading'],
            background=self.theme.COLORS['light_gray'], foreground=self.theme.COLORS['dark_gray']
        )
        title_label.grid(row=0, column=0, sticky='w', pady=(0, self.theme.SPACING['sm']))

        self.results_text = ModernScrolledText(self, height=8, wrap=tk.WORD, font=self.theme.FONTS['monospace'])
        self.results_text.grid(row=1, column=0, sticky='nsew')
        self.results_text.configure_tags()

        self.show_welcome_message()

    def show_welcome_message(self):
        """Display welcome instructions."""
        welcome_text = """🎉 Welcome to Email Spam Detector!

1. Paste a message above
2. Click 'Analyze Message'
3. Try the sample messages

Ready to detect spam! 🚀"""
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", welcome_text, "info")

    def show_analysis_result(self, result, original_message):
        """Display analysis results or errors."""
        self.results_text.delete("1.0", tk.END)
        if result.get('error'):
            self._show_error_result(result)
        else:
            self._show_success_result(result, original_message)

    def _show_success_result(self, result, original_message):
        is_spam = result.get('is_spam', False)
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        spam_prob = result.get('spam_probability', 0)

        header = "🔍 ANALYSIS COMPLETE\n" + "="*50 + "\n\n"
        self.results_text.insert(tk.END, header, "info")

        tag = "spam" if is_spam else "ham"
        self.results_text.insert(tk.END, f"Result: {prediction}\n", tag)

        confidence_text = f"\nConfidence: {confidence:.1%}\nSpam Probability: {spam_prob:.1%}\nHam Probability: {1-spam_prob:.1%}\n\n"
        self.results_text.insert(tk.END, confidence_text, "info")

        preview = original_message[:100] + "..." if len(original_message) > 100 else original_message
        self.results_text.insert(tk.END, f"Analyzed Message:\n\"{preview}\"\n\n", "monospace")

        interpretation = (
            "🚨 This message appears to be SPAM.\n• Urgent language\n• Suspicious links"
            if is_spam else
            "✅ This message appears to be legitimate (HAM).\nNormal communication detected."
        )
        self.results_text.insert(tk.END, interpretation, tag)

    def _show_error_result(self, result):
        self.results_text.insert("1.0", f"❌ Analysis Error:\n{result.get('prediction', 'Unknown error')}", "warning")

    def show_loading_state(self):
        self.results_text.delete("1.0", tk.END)
        loading_text = "🔄 Analyzing message...\nPlease wait while the AI model processes your message."
        self.results_text.insert("1.0", loading_text, "info")
