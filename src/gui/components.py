import os
import tkinter as tk
from tkinter import ttk

import pandas as pd

from .styles import ModernScrolledText, SpamDetectorTheme


class MessageInputPanel(tk.Frame):
    """Reusable input panel for pasting and analyzing messages."""

    def __init__(self, parent, on_analyze_callback=None, **kwargs):
        super().__init__(
            parent,
            **kwargs,
            padx=SpamDetectorTheme.SPACING['lg'],
            pady=SpamDetectorTheme.SPACING['lg'],
            highlightbackground=SpamDetectorTheme.COLORS['border'],
            highlightthickness=1,
        )
        self.theme = SpamDetectorTheme()
        self.on_analyze_callback = on_analyze_callback
        self.placeholder_text = "Paste an email, SMS, or message here..."

        self._load_sample_data()
        self.create_widgets()
        self.setup_placeholder()

    def _load_sample_data(self):
        """Load sample data from CSV for demo buttons."""
        try:
            sample_path = 'data/samples/sample.csv'
            if os.path.exists(sample_path):
                self.sample_data = pd.read_csv(sample_path)
                self.ham_samples = self.sample_data[self.sample_data['label'] == 0]
                self.spam_samples = self.sample_data[self.sample_data['label'] == 1]
            else:
                self.sample_data = None
                self.ham_samples = None
                self.spam_samples = None
        except Exception:
            self.sample_data = None
            self.ham_samples = None
            self.spam_samples = None

    def create_widgets(self):
        """Create widgets: title, text area, buttons, sample messages."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        title_row = tk.Frame(self, background=self.theme.COLORS['surface'])
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, self.theme.SPACING['md']))
        title_row.columnconfigure(0, weight=1)

        tk.Label(
            title_row,
            text="Message Input",
            font=self.theme.FONTS['heading'],
            background=self.theme.COLORS['surface'],
            foreground=self.theme.COLORS['text'],
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            title_row,
            text="Paste content or load a sample",
            font=self.theme.FONTS['small'],
            background=self.theme.COLORS['surface'],
            foreground=self.theme.COLORS['text_muted'],
        ).grid(row=0, column=1, sticky="e")

        self.text_input = ModernScrolledText(self, height=8, wrap=tk.WORD, font=self.theme.FONTS['body'])
        self.text_input.grid(row=1, column=0, sticky="nsew", pady=(0, self.theme.SPACING['md']))
        self.text_input.configure_tags()

        button_frame = tk.Frame(self, background=self.theme.COLORS['surface'])
        button_frame.grid(row=2, column=0, sticky="ew")
        button_frame.columnconfigure(2, weight=1)

        self.analyze_button = ttk.Button(
            button_frame,
            text="Analyze Message",
            style="Modern.TButton",
            command=self.on_analyze_clicked,
            state=tk.DISABLED,
        )
        self.analyze_button.grid(row=0, column=0, padx=(0, self.theme.SPACING['sm']))

        self.clear_button = ttk.Button(
            button_frame,
            text="Clear",
            style="Danger.TButton",
            command=self.clear_text,
        )
        self.clear_button.grid(row=0, column=1, padx=(0, self.theme.SPACING['md']))

        sample_frame = tk.Frame(button_frame, background=self.theme.COLORS['surface'])
        sample_frame.grid(row=0, column=3, sticky="e")

        ttk.Button(
            sample_frame,
            text="Ham Sample",
            style="Secondary.TButton",
            command=self.load_ham_sample,
        ).pack(side=tk.LEFT, padx=(0, self.theme.SPACING['xs']))

        ttk.Button(
            sample_frame,
            text="Spam Sample",
            style="Secondary.TButton",
            command=self.load_spam_sample,
        ).pack(side=tk.LEFT)

    def setup_placeholder(self):
        """Insert placeholder and bind events."""
        self.text_input.insert("1.0", self.placeholder_text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['text_subtle'])
        self.text_input.text_widget.bind("<FocusIn>", self.on_focus_in)
        self.text_input.text_widget.bind("<FocusOut>", self.on_focus_out)
        self.text_input.text_widget.bind("<KeyRelease>", self.on_text_change)

    def on_focus_in(self, event):
        """Remove placeholder on focus."""
        if self.get_text() == self.placeholder_text:
            self.text_input.delete("1.0", tk.END)
            self.text_input.text_widget.configure(foreground=self.theme.COLORS['text'])

    def on_focus_out(self, event):
        """Restore placeholder if empty."""
        if not self.get_text():
            self.text_input.insert("1.0", self.placeholder_text)
            self.text_input.text_widget.configure(foreground=self.theme.COLORS['text_subtle'])

    def on_text_change(self, event):
        """Enable/disable analyze button based on text content."""
        text = self.get_text()
        has_content = text and text != self.placeholder_text
        self.analyze_button.configure(state=tk.NORMAL if has_content else tk.DISABLED)

    def on_analyze_clicked(self):
        """Invoke callback on analyze."""
        if self.on_analyze_callback:
            text = self.get_text()
            if text and text != self.placeholder_text:
                self.on_analyze_callback(text)

    def get_text(self):
        """Get current text from input area."""
        return self.text_input.get("1.0", tk.END).strip()

    def set_text(self, text):
        """Set text in input area."""
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['text'])
        self.on_text_change(None)

    def clear_text(self):
        """Clear text and restore placeholder."""
        self.set_text(self.placeholder_text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['text_subtle'])
        self.analyze_button.configure(state=tk.DISABLED)

    def load_ham_sample(self):
        """Load a random ham sample from sample.csv."""
        if self.ham_samples is None or len(self.ham_samples) == 0:
            self.set_text("Hi, are we still meeting tomorrow morning? I can bring the documents.")
            return

        try:
            random_ham = self.ham_samples.sample(n=1).iloc[0]
            self.set_text(random_ham['processed_message'])
        except Exception:
            self.set_text("Hi, are we still meeting tomorrow morning? I can bring the documents.")

    def load_spam_sample(self):
        """Load a random spam sample from sample.csv."""
        if self.spam_samples is None or len(self.spam_samples) == 0:
            self.set_text("Congratulations! You have won a cash prize. Claim now by clicking this link.")
            return

        try:
            random_spam = self.spam_samples.sample(n=1).iloc[0]
            self.set_text(random_spam['processed_message'])
        except Exception:
            self.set_text("Congratulations! You have won a cash prize. Claim now by clicking this link.")


class ResultsDisplayPanel(tk.Frame):
    """Reusable panel to display spam analysis results."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            **kwargs,
            padx=SpamDetectorTheme.SPACING['lg'],
            pady=SpamDetectorTheme.SPACING['lg'],
            highlightbackground=SpamDetectorTheme.COLORS['border'],
            highlightthickness=1,
        )
        self.theme = SpamDetectorTheme()
        self.create_widgets()

    def create_widgets(self):
        """Create results display widgets."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        title_row = tk.Frame(self, background=self.theme.COLORS['surface'])
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, self.theme.SPACING['md']))
        title_row.columnconfigure(0, weight=1)

        tk.Label(
            title_row,
            text="Analysis Results",
            font=self.theme.FONTS['heading'],
            background=self.theme.COLORS['surface'],
            foreground=self.theme.COLORS['text'],
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            title_row,
            text="Classification and confidence",
            font=self.theme.FONTS['small'],
            background=self.theme.COLORS['surface'],
            foreground=self.theme.COLORS['text_muted'],
        ).grid(row=0, column=1, sticky="e")

        self.results_text = ModernScrolledText(self, height=8, wrap=tk.WORD, font=self.theme.FONTS['monospace'])
        self.results_text.grid(row=1, column=0, sticky="nsew")
        self.results_text.configure_tags()

        self.show_welcome_message()

    def show_welcome_message(self):
        """Display welcome instructions."""
        welcome_text = """Ready for analysis

1. Paste or load a sample message
2. Run the classifier
3. Review the label, confidence, and risk notes

The model output will appear here."""

        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", welcome_text, "info")

    def show_analysis_result(self, result, original_message):
        """Display analysis results or errors."""
        self.results_text.delete("1.0", tk.END)

        if result.get('error'):
            self.show_error_result(result)
        else:
            self.show_success_result(result, original_message)

    def show_success_result(self, result, original_message):
        """Display successful analysis results."""
        is_spam = result.get('is_spam', False)
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        spam_prob = result.get('spam_probability', 0)

        tag = "spam" if is_spam else "ham"
        verdict = "SPAM DETECTED" if is_spam else "LIKELY LEGITIMATE"

        self.results_text.insert(tk.END, "ANALYSIS COMPLETE\n", "info")
        self.results_text.insert(tk.END, "-" * 52 + "\n\n", "info")
        self.results_text.insert(tk.END, f"{verdict}\n", tag)
        self.results_text.insert(tk.END, f"Classifier label: {prediction}\n\n", tag)

        confidence_text = f"""Confidence:       {confidence:.1%}
Spam probability: {spam_prob:.1%}
Ham probability:  {(1 - spam_prob):.1%}

"""
        self.results_text.insert(tk.END, confidence_text, "info")

        preview = original_message[:160] + "..." if len(original_message) > 160 else original_message
        self.results_text.insert(tk.END, "Message preview\n", "info")
        self.results_text.insert(tk.END, f"{preview}\n\n", "monospace")

        interpretation = (
            "Recommended action: Treat this message with caution. Watch for urgency, links, "
            "attachments, and requests for personal or financial information."
            if is_spam else
            "Recommended action: This message looks consistent with normal communication. "
            "Still verify links or unusual requests before taking action."
        )
        self.results_text.insert(tk.END, interpretation, tag)

    def show_error_result(self, result):
        """Display error message."""
        error_text = f"Analysis Error\n\n{result.get('prediction') or result.get('error') or 'Unknown error'}"
        self.results_text.insert("1.0", error_text, "warning")

    def show_loading_state(self):
        """Show loading message during analysis."""
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(
            "1.0",
            "Analyzing message...\n\nPlease wait while the model processes the text.",
            "info",
        )
