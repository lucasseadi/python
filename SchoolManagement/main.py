from tkinter import *
from front import SchoolManagement

if __name__ == "__main__":
    root = Tk()
    school = SchoolManagement(root)
    root.protocol("WM_DELETE_WINDOW", school.on_close_window)
    root.mainloop()
