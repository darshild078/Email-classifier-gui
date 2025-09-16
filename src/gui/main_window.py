"""
Main application window with ML-powered spam detection.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .styles import apply_modern_style, SpamDetectorTheme, create_status_indicator
from .components import MessageInputPanel, ResultsDisplayPanel
from core.predictor import SpamPredictionEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SpamDetectorMainWindow:
    """Main GUI with spam detection, async model loading, and error handling."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.theme = SpamDetectorTheme()
        self.prediction_engine = None
        self.is_model_loading = False
        self.is_analyzing = False
        self._setup_window()
        self._apply_styling()
        self._create_widgets()
        self._start_model_loading()
        logger.info("Main window initialized")
    
    def _setup_window(self):
        """Configure window basics."""
        self.root.title("🚨 Email Spam Detector - Professional AI Tool")
        self.root.geometry("900x700")
        self.root.minsize(self.theme.SIZES['window_min_width'], self.theme.SIZES['window_min_height'])
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"900x700+{x}+{y}")
        self.root.configure(background=self.theme.COLORS['light_gray'])
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _apply_styling(self):
        """Apply modern style."""
        self.style, _ = apply_modern_style()
    
    def _create_widgets(self):
        """Build app UI."""
        main_container = tk.Frame(self.root, background=self.theme.COLORS['light_gray'],
                                  padx=self.theme.SPACING['xl'], pady=self.theme.SPACING['lg'])
        main_container.pack(fill=tk.BOTH, expand=True)
        self._create_header(main_container)
        self._create_status_section(main_container)
        self._create_main_content(main_container)
        self._create_footer(main_container)
    
    def _create_header(self, parent):
        """Header with title/subtitle."""
        header_frame = tk.Frame(parent, background=self.theme.COLORS['light_gray'])
        header_frame.pack(fill=tk.X, pady=(0, self.theme.SPACING['lg']))
        tk.Label(header_frame, text="🚨 Email Spam Detector", font=self.theme.FONTS['title'],
                 background=self.theme.COLORS['light_gray'], foreground=self.theme.COLORS['primary']).pack()
        tk.Label(header_frame, text="AI-Powered Message Analysis Tool", font=self.theme.FONTS['body'],
                 background=self.theme.COLORS['light_gray'], foreground=self.theme.COLORS['medium_gray']).pack(
                     pady=(self.theme.SPACING['xs'], 0))
    
    def _create_status_section(self, parent):
        """Status area with indicator and model info."""
        status_frame = tk.Frame(parent, background=self.theme.COLORS['light_gray'])
        status_frame.pack(fill=tk.X, pady=(0, self.theme.SPACING['md']))
        self.status_indicator, self.status_label = create_status_indicator(
            status_frame, "🔄 Loading AI model...", "info"
        )
        self.status_indicator.pack(side=tk.LEFT)
        self.info_button = ttk.Button(status_frame, text="ℹ️ Model Info", command=self._show_model_info,
                                      state=tk.DISABLED)
        self.info_button.pack(side=tk.RIGHT)
    
    def _create_main_content(self, parent):
        """Resizable content area with input/results panels."""
        paned_window = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=(0, self.theme.SPACING['md']))
        input_frame = tk.Frame(paned_window, background=self.theme.COLORS['light_gray'])
        self.message_input = MessageInputPanel(input_frame, on_analyze_callback=self._analyze_message,
                                               background=self.theme.COLORS['light_gray'])
        self.message_input.pack(fill=tk.BOTH, expand=True, padx=self.theme.SPACING['md'])
        paned_window.add(input_frame, weight=1)
        results_frame = tk.Frame(paned_window, background=self.theme.COLORS['light_gray'])
        self.results_display = ResultsDisplayPanel(results_frame, background=self.theme.COLORS['light_gray'])
        self.results_display.pack(fill=tk.BOTH, expand=True, padx=self.theme.SPACING['md'])
        paned_window.add(results_frame, weight=1)
    
    def _create_footer(self, parent):
        """Footer with tips/version."""
        footer_frame = tk.Frame(parent, background=self.theme.COLORS['light_gray'])
        footer_frame.pack(fill=tk.X, pady=(self.theme.SPACING['md'], 0))
        tk.Label(footer_frame, text="💡 Tip: Paste emails, SMS, or text to check for spam",
                 font=self.theme.FONTS['small'], background=self.theme.COLORS['light_gray'],
                 foreground=self.theme.COLORS['medium_gray']).pack(side=tk.LEFT)
        tk.Label(footer_frame, text="v1.0.0", font=self.theme.FONTS['small'],
                 background=self.theme.COLORS['light_gray'],
                 foreground=self.theme.COLORS['medium_gray']).pack(side=tk.RIGHT)
    
    def _start_model_loading(self):
        """Load models in background."""
        if self.is_model_loading: return
        self.is_model_loading = True
        def load_models():
            try:
                self.prediction_engine = SpamPredictionEngine()
                success = self.prediction_engine.load_models()
                self.root.after(0, lambda: self._on_model_loading_complete(success))
            except Exception as e:
                logger.error(f"Model load failed: {e}")
                self.root.after(0, lambda: self._on_model_loading_complete(False, str(e)))
        threading.Thread(target=load_models, daemon=True).start()
        logger.info("Model loading started")
    
    def _on_model_loading_complete(self, success, error=None):
        """Handle model load results."""
        self.is_model_loading = False
        if success:
            self.status_label.configure(text="✅ AI model ready - Start analyzing!")
            self.status_indicator.configure(background=self.theme.COLORS['light_gray'])
            self.message_input.analyze_button.configure(state=tk.NORMAL)
            self.info_button.configure(state=tk.NORMAL)
            logger.info("Model loaded successfully")
        else:
            self.status_label.configure(text=f"❌ Model loading failed: {error or ''}")
            self.results_display.results_text.delete("1.0", tk.END)
            self.results_display.results_text.insert(
                "1.0",
                f"""⚠️  Model Not Available

Ensure dataset in data/raw/spam.csv
Run: python train_model.py
Restart app

Error: {error or 'Unknown'}""",
                "warning"
            )
            logger.error(f"Model loading failed: {error}")
    
    def _analyze_message(self, message):
        """Run spam analysis."""
        if not self.prediction_engine or not self.prediction_engine.is_ready:
            messagebox.showerror("Error", "AI model not ready yet.")
            return
        if self.is_analyzing: return
        self.is_analyzing = True
        self.message_input.analyze_button.configure(text="🔄 Analyzing...", state=tk.DISABLED)
        self.results_display.show_loading_state()
        def analyze():
            try:
                result = self.prediction_engine.predict_message(message)
                self.root.after(0, lambda: self._on_analysis_complete(result, message))
            except Exception as e:
                self.root.after(0, lambda: self._on_analysis_complete(
                    {'success': False, 'error': f'Analysis failed: {e}', 'prediction': None}, message))
        threading.Thread(target=analyze, daemon=True).start()
    
    def _on_analysis_complete(self, result, original_message):
        """Show analysis result."""
        self.is_analyzing = False
        self.message_input.analyze_button.configure(text="🔍 Analyze Message", state=tk.NORMAL)
        self.results_display.show_analysis_result(result, original_message)
        if result.get('success'):
            logger.info(f"Analysis done: {result.get('prediction')} ({result.get('confidence', 0):.3f})")
        else:
            logger.warning(f"Analysis failed: {result.get('error')}")
    
    def _show_model_info(self):
        """Show model details dialog."""
        if not self.prediction_engine or not self.prediction_engine.is_ready:
            messagebox.showwarning("Warning", "Model not ready")
            return
        try:
            info = self.prediction_engine.get_model_info()
            text = f"""🤖 Model Info

Type: {info.get('model_type', 'Unknown')}
Features: {info.get('feature_count', 0):,}
Status: {'Ready' if info.get('model_ready') else 'Not Ready'}

Preprocessing:"""
            for step in info.get('preprocessing_steps', []):
                text += f"\n  • {step}"
            text += "\n\nUses Naive Bayes + TF-IDF for spam detection."
            messagebox.showinfo("Model Information", text)
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve info: {e}")
    
    def _on_closing(self):
        """Graceful exit."""
        logger.info("Closing app")
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Run main loop."""
        logger.info("Starting app loop")
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"App error: {e}")
            messagebox.showerror("App Error", f"Unexpected error: {e}")
        finally:
            logger.info("App terminated")
