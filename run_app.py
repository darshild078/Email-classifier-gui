"""
Launcher for Email Spam Detector GUI.
"""
import sys
import os
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import traceback

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 8):
        return False, f"Python 3.8+ required. Current: {sys.version}"
    return True, "Python version OK"

def check_dependencies():
    """Check required packages."""
    required = [
        ('tkinter', 'tkinter'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('nltk', 'nltk'),
        ('pickle', 'pickle (built-in)')
    ]
    missing = []
    for pkg, name in required:
        try:
            if pkg == 'sklearn':
                import sklearn
            else:
                __import__(pkg)
        except ImportError:
            missing.append(name)
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "All dependencies OK"

def check_model_files():
    """Check model files exist."""
    files = ['models/spam_classifier.pkl', 'models/feature_engineer.pkl']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return False, f"Missing files: {', '.join(missing)}"
    return True, "Model files found"

def check_data_directory():
    """Check/create required dirs."""
    dirs = ['data', 'models', 'logs']
    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create {d}: {e}"
    return True, "Directories OK"

def show_setup_dialog():
    """Show setup guide."""
    root = tk.Tk(); root.withdraw()
    msg = """🚨 Email Spam Detector - Setup Needed

Some components are missing:

📋 Steps:
1️⃣ Install deps → pip install -r requirements.txt
2️⃣ Get dataset → spam.csv in data/raw/
3️⃣ Train model → python train_model.py
4️⃣ Run app → python run_app.py

Open training script now?"""
    if messagebox.askyesno("Setup Required", msg):
        try:
            import subprocess
            subprocess.run([sys.executable, "train_model.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Could not run training: {e}")
    root.destroy()

def run_application():
    """Start GUI app."""
    try:
        from gui.main_window import SpamDetectorMainWindow
        app = SpamDetectorMainWindow()
        app.run()
    except ImportError as e:
        messagebox.showerror("Import Error",
                             f"Import error: {e}\nCheck file structure.")
        return False
    except Exception as e:
        messagebox.showerror("App Error",
                             f"Error: {e}\n\n{traceback.format_exc()}")
        return False
    return True

def main():
    """Main entry with validation."""
    print("🚀 Starting Email Spam Detector...")
    print("=" * 50)
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_data_directory),
        ("Model Files", check_model_files)
    ]
    all_ok, failed = True, []
    for name, fn in checks:
        try:
            ok, msg = fn()
            status = "✅" if ok else "❌"
            print(f"{status} {name}: {msg}")
            if not ok: all_ok, failed = False, failed + [name]
        except Exception as e:
            print(f"❌ {name}: error - {e}")
            all_ok, failed = False, failed + [name]
    print("-" * 50)
    if all_ok:
        print("✅ All checks passed. Launching GUI...")
        if run_application():
            print("👋 Closed normally")
        else:
            print("❌ Application error")
            return 1
    else:
        print(f"❌ Setup incomplete. Failed: {', '.join(failed)}")
        try:
            show_setup_dialog()
        except Exception as e:
            print(f"Could not show setup dialog: {e}")
            print("Run 'python train_model.py' first")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
