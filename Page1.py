import tkinter as tk
from tkinter import ttk

root = tk.Tk()
mode = None  # Global variable to store the selected mode

def go_static():
    global mode
    mode = "S"
    for widget in root.winfo_children():
        widget.destroy()
    
def go_dynamic():
    global mode
    mode = "D"
    for widget in root.winfo_children():
        widget.destroy()
    
try:
    root.state('zoomed')  # This works on Windows
except Exception:
    try:
        root.attributes('-fullscreen', True)  # This works on macOS and Linux
    except Exception:
        pass

def ShowMainPage():
    """Return to the main page."""
    for widget in root.winfo_children():
        widget.destroy()
    MainPage()

def ShowFirstPage():
    for widget in root.winfo_children():
        widget.destroy()
    root.configure(bg="black")
    root.title("Path Planning")
    # window is already maximized; do not force a fixed geometry
    root.update_idletasks()
    title_label = tk.Label(root, text="Path Planning", bg="black", fg="white", font=("Bahnschrift", 36))
    title_label.pack(pady=50)
    start_button = tk.Button(
        root,
        text="Start",
        bg="#222222",
        fg="white",
        activebackground="#444444",
        activeforeground="white",
        font=("Bahnschrift", 24),
        width=10,
        height=2,
        borderwidth=0,
        highlightthickness=0,
        # open the main mode selection page first (don't show SecondPage here)
        command=lambda: MainPage()
    )
    start_button.pack(pady=30)
    credit_label = tk.Label(root, text="Created by Ahmed Yacine Ahriche", bg="black", fg="white", font=("Bahnschrift", 10))
    credit_label.pack(side="bottom", pady=5)
    close_button = tk.Button(
        root,
        text="Close",
        command=root.destroy,
        bg="#222222",
        fg="white",
        activebackground="#444444",
        activeforeground="white",
        font=("Bahnschrift", 12),
        width=10,
        height=2,
        borderwidth=0,
        highlightthickness=0
    )
    close_button.pack(side="bottom", pady=10)

def MainPage():
    """Show the mode selection label and 2D/3D buttons."""
    for widget in root.winfo_children():
        widget.destroy()
    root.configure(bg="black")
    root.title("Path Planning")
    root.update_idletasks()

    mode_label = tk.Label(root, text="Choose your MODE", bg="black", fg="white", font=("Bahnschrift", 36))
    mode_label.pack(pady=50)

    button_frame = tk.Frame(root, bg="black")
    button_frame.pack(pady=20)

    def lunch_static2d():
        from Static2D import static2d
        static2d(root,MainPage)

    def lunch_dynamic2d():
        from Dynamic2D import dynamic2d
        dynamic2d(root,MainPage)
    
    def static_button_action():
        go_static()         # cleans window and sets mode
        lunch_static2d()    # launches your Static2D module
    
    btn_dy =tk.Button(
        button_frame,
        text="Dynamic",
        bg="#222222",
        fg="white",
        activebackground="#444444",
        activeforeground="white",
        font=("Bahnschrift", 24),
        width=10,
        height=2,
        borderwidth=0,
        highlightthickness=0,
        command= lunch_dynamic2d,
        )
    btn_dy.pack(side=tk.LEFT, padx=20)

    btn_st =tk.Button(
        button_frame,
        text="Static",
        bg="#222222",
        fg="white",
        activebackground="#444444",
        activeforeground="white",
        font=("Bahnschrift", 24),
        width=10,
        height=2,
        borderwidth=0,
        highlightthickness=0,
        command= static_button_action,
        )
    btn_st.pack(side=tk.LEFT, padx=20)
   

if __name__ == "__main__":
    ShowFirstPage()
    root.mainloop()