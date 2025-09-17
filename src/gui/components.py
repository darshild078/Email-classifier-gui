import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
from .styles import SpamDetectorTheme, ModernScrolledText, create_status_indicator

class MessageInputPanel(tk.Frame):
    """Reusable input panel for pasting and analyzing messages."""
    
    def __init__(self, parent, on_analyze_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.theme = SpamDetectorTheme()
        self.on_analyze_callback = on_analyze_callback
        self.placeholder_text = "Paste your message here to check if it's spam..."
        
        # Load sample data for demo buttons
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
                print(f"✅ Loaded {len(self.ham_samples)} ham and {len(self.spam_samples)} spam samples")
            else:
                print("⚠️ Sample data not found at data/samples/sample.csv")
                self.sample_data = None
                self.ham_samples = None
                self.spam_samples = None
        except Exception as e:
            print(f"⚠️ Error loading sample data: {e}")
            self.sample_data = None
            self.ham_samples = None
            self.spam_samples = None

    def create_widgets(self):
        """Create widgets: title, text area, buttons, sample messages."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Title label
        title_label = tk.Label(
            self, text="Message Input", 
            font=self.theme.FONTS['heading'],
            background=self.theme.COLORS['light_gray'],
            foreground=self.theme.COLORS['dark_gray']
        )
        title_label.grid(row=0, column=0, sticky="w", pady=(0, self.theme.SPACING['sm']))
        
        # Text input area
        self.text_input = ModernScrolledText(
            self, height=8, wrap=tk.WORD, 
            font=self.theme.FONTS['body']
        )
        self.text_input.grid(row=1, column=0, sticky="nsew", pady=(0, self.theme.SPACING['md']))
        self.text_input.configure_tags()
        
        # Buttons frame
        button_frame = tk.Frame(self, background=self.theme.COLORS['light_gray'])
        button_frame.grid(row=2, column=0, sticky="ew")
        button_frame.columnconfigure(1, weight=1)
        
        # Analyze button
        self.analyze_button = ttk.Button(
            button_frame, text="🔍 Analyze Message", 
            style="Modern.TButton",
            command=self.on_analyze_clicked,
            state=tk.DISABLED
        )
        self.analyze_button.grid(row=0, column=0, padx=(0, self.theme.SPACING['sm']))
        
        # Clear button
        self.clear_button = ttk.Button(
            button_frame, text="🗑️ Clear", 
            style="Danger.TButton",
            command=self.clear_text
        )
        self.clear_button.grid(row=0, column=2, sticky="e")
        
        # Sample buttons frame
        sample_frame = tk.Frame(button_frame, background=self.theme.COLORS['light_gray'])
        sample_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(self.theme.SPACING['sm'], 0))
        
        tk.Label(
            sample_frame, text="Try samples:", 
            font=self.theme.FONTS['small'],
            background=self.theme.COLORS['light_gray'],
            foreground=self.theme.COLORS['medium_gray']
        ).pack(side=tk.LEFT, padx=(0, self.theme.SPACING['sm']))
        
        ttk.Button(
            sample_frame, text="📧 Ham Example", 
            command=self.load_ham_sample
        ).pack(side=tk.LEFT, padx=(0, self.theme.SPACING['xs']))
        
        ttk.Button(
            sample_frame, text="🚨 Spam Example", 
            command=self.load_spam_sample
        ).pack(side=tk.LEFT)

    def setup_placeholder(self):
        """Insert placeholder and bind events."""
        self.text_input.insert("1.0", self.placeholder_text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['medium_gray'])
        self.text_input.text_widget.bind("<FocusIn>", self.on_focus_in)
        self.text_input.text_widget.bind("<FocusOut>", self.on_focus_out)
        self.text_input.text_widget.bind("<KeyRelease>", self.on_text_change)

    def on_focus_in(self, event):
        """Remove placeholder on focus."""
        if self.get_text() == self.placeholder_text:
            self.text_input.delete("1.0", tk.END)
            self.text_input.text_widget.configure(foreground=self.theme.COLORS['dark_gray'])

    def on_focus_out(self, event):
        """Restore placeholder if empty."""
        if not self.get_text():
            self.text_input.insert("1.0", self.placeholder_text)
            self.text_input.text_widget.configure(foreground=self.theme.COLORS['medium_gray'])

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
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['dark_gray'])
        self.on_text_change(None)

    def clear_text(self):
        """Clear text and restore placeholder."""
        self.set_text(self.placeholder_text)
        self.text_input.text_widget.configure(foreground=self.theme.COLORS['medium_gray'])
        self.analyze_button.configure(state=tk.DISABLED)

    def load_ham_sample(self):
        """Load a random ham sample from sample.csv."""
        if self.ham_samples is None or len(self.ham_samples) == 0:
            # Fallback to hardcoded sample if no data available
            sample = "Hi! Let's meet tomorrow for coffee. Looking forward to catching up!"
            self.set_text(sample)
            print("⚠️ Using fallback ham sample - sample.csv not available")
            return
        
        try:
            # Get random ham message
            random_ham = self.ham_samples.sample(n=1).iloc[0]
            message = random_ham['processed_message']
            self.set_text(message)
            print("✅ Loaded random ham sample from data")
        except Exception as e:
            # Fallback if error occurs
            sample = "Hi! Let's meet tomorrow for coffee. Looking forward to catching up!"
            self.set_text(sample)
            print(f"⚠️ Error loading ham sample, using fallback: {e}")

    def load_spam_sample(self):
        """Load a random spam sample from sample.csv."""
        if self.spam_samples is None or len(self.spam_samples) == 0:
            # Fallback to hardcoded sample if no data available
            sample = "CONGRATULATIONS! You've WON $5000 CASH PRIZE! Click here NOW!"
            self.set_text(sample)
            print("⚠️ Using fallback spam sample - sample.csv not available")
            return
        
        try:
            # Get random spam message
            random_spam = self.spam_samples.sample(n=1).iloc[0]
            message = random_spam['processed_message']
            self.set_text(message)
            print("✅ Loaded random spam sample from data")
        except Exception as e:
            # Fallback if error occurs
            sample = "CONGRATULATIONS! You've WON $5000 CASH PRIZE! Click here NOW!"
            self.set_text(sample)
            print(f"⚠️ Error loading spam sample, using fallback: {e}")


class ResultsDisplayPanel(tk.Frame):
    """Reusable panel to display spam analysis results."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.theme = SpamDetectorTheme()
        self.create_widgets()

    def create_widgets(self):
        """Create results display widgets."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Title label
        title_label = tk.Label(
            self, text="Analysis Results", 
            font=self.theme.FONTS['heading'],
            background=self.theme.COLORS['light_gray'],
            foreground=self.theme.COLORS['dark_gray']
        )
        title_label.grid(row=0, column=0, sticky="w", pady=(0, self.theme.SPACING['sm']))
        
        # Results text area
        self.results_text = ModernScrolledText(
            self, height=8, wrap=tk.WORD, 
            font=self.theme.FONTS['monospace']
        )
        self.results_text.grid(row=1, column=0, sticky="nsew")
        self.results_text.configure_tags()
        
        self.show_welcome_message()

    def show_welcome_message(self):
        """Display welcome instructions."""
        welcome_text = """🚀 Welcome to Email Spam Detector!

📝 How to use:
1. Paste a message above
2. Click 'Analyze Message'  
3. Try the sample messages

🤖 Ready to detect spam!"""
        
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
        
        # Header
        header = f"🔍 ANALYSIS COMPLETE\n{'='*50}\n"
        self.results_text.insert(tk.END, header, "info")
        
        # Result with emoji and color
        tag = "spam" if is_spam else "ham"
        emoji = "🚨" if is_spam else "✅"
        self.results_text.insert(tk.END, f"{emoji} Result: {prediction}\n", tag)
        
        # Confidence details
        confidence_text = f"""
📊 Confidence: {confidence:.1%}
📈 Spam Probability: {spam_prob:.1%}
📉 Ham Probability: {(1-spam_prob):.1%}

"""
        self.results_text.insert(tk.END, confidence_text, "info")
        
        # Message preview
        preview = original_message[:100] + "..." if len(original_message) > 100 else original_message
        self.results_text.insert(tk.END, f"📝 Analyzed Message:\n{preview}\n\n", "monospace")
        
        # Interpretation
        interpretation = (
            "⚠️ This message appears to be SPAM. Exercise caution with:\n• Urgent language\n• Suspicious links\n• Requests for personal info"
            if is_spam else
            "✅ This message appears to be legitimate HAM.\n• Normal communication detected\n• No suspicious patterns found"
        )
        self.results_text.insert(tk.END, interpretation, tag)

    def show_error_result(self, result):
        """Display error message."""
        error_text = f"❌ Analysis Error\n\n{result.get('prediction', 'Unknown error')}"
        self.results_text.insert("1.0", error_text, "warning")

    def show_loading_state(self):
        """Show loading message during analysis."""
        self.results_text.delete("1.0", tk.END)
        loading_text = "🔄 Analyzing message...\n\nPlease wait while the AI model processes your message."
        self.results_text.insert("1.0", loading_text, "info")
