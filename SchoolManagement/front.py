import hashlib
import re
from back import Back
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter.ttk import Notebook


class SchoolManagement(Tk):
    resultStr = ""

    def __init__(self, master, data_query=Back):
        try:
            self.data_query = data_query
            self.root = master
            self.main_menu = None
            self.back = Back()
            self.login_screen()
        except Exception as e:
            print("Error! ", e)
            self.root.destroy()
            exit(0)

    def login_screen(self):
        # Main function
        self.root.geometry("300x200")
        self.root.title("School Login")
        self.root.resizable(0, 0)

        self.main_menu = Frame()
        self.main_menu.place(relwidth=.9, relheight=.9, relx=.05, rely=.03)

        username_label = Label(self.main_menu, text="Username:")
        self.username_entry = Entry(self.main_menu, width=30)
        username_label.place(relx=.005, rely=.3)
        self.username_entry.place(relx=.3, rely=.3)

        password_label = Label(self.main_menu, text="Password:")
        self.password_entry = Entry(self.main_menu, width=30, show="*")
        password_label.place(relx=.005, rely=.45)
        self.password_entry.place(relx=.3, rely=.45)

        enter = Button(self.main_menu, text="Enter", fg="green",
                       command=lambda: self.authenticate(self.username_entry.get(), self.password_entry.get()))
        self.root.bind('<Return>', self.parseEnter)
        enter.place(relx=.45, rely=.7)

    def parseEnter(self, event):
        self.authenticate(self.username_entry.get(), self.password_entry.get())

    def authenticate(self, username, password):
        result_of_login = StringVar()
        result_label = Label(self.main_menu, textvariable=result_of_login)
        result_label.place(relx=.3, rely=.9)

        # Check for empty database
        db_count = self.back.check_empty_database()
        if db_count[0][0] == 0:
            print("Empty database.")
            result_of_login.set("Empty database.")
            return
        else:
            matched_username = False
            matched_password = False

            # Convert the user input to hashes
            hashed_username = hashlib.md5(username.encode("UTF-8")).hexdigest()
            hashed_password = hashlib.md5(password.encode("UTF-8")).hexdigest()

            # Select all usernames to compare user input
            query = self.back.select("SELECT USERNAME FROM LOGINCREDENTIALS")
            length = len(query)

            for i in range(length):
                db_username = query[i][0]
                if db_username == hashed_username:
                    matched_username = True
                # Make sure the username and password are from the same tuple
                id_num = self.back.select("SELECT USERID FROM LOGINCREDENTIALS WHERE USERNAME = ?",
                                          (hashed_username,))
            if matched_username == TRUE:
                length = len(id_num)

                for i in range(length):
                    query = self.back.select("SELECT PASSWORD FROM LOGINCREDENTIALS WHERE USERID = ?",
                                             (id_num[i][0],))

                    # Compare hash to db value
                    if query[i][0] == hashed_password:
                        matched_password = True

            if matched_username is True and matched_password is True:
                result_of_login.set("Login successful")
                self.login_successful()
                return
            else:
                result_of_login.set("Incorrect credentials.")
                return

    # Function for hiding something from the screen
    def forget(self, widget):
        widget.place_forget()

    # Changes from login screen into management screen
    def login_successful(self):
        self.forget(self.main_menu)

        self.root.title("School Management")
        self.root.minsize(850, 600)
        self.root.config(padx=10)
        self.root.geometry("850x600")
        self.root.resizable(width=False, height=False)

        self.notebook = Notebook()

        # Add course tab
        self.add_course_tab = Frame(self.notebook, bg="#42bcf5")
        self.create_left_icon_courses(self.add_course_tab)
        self.create_label_frame_courses(self.add_course_tab)
        self.create_message_area_courses(self.add_course_tab)
        self.create_tree_view_course()
        self.create_scrollbar_courses(self.add_course_tab, self.tree_courses)
        self.create_bottom_buttons_courses(self.add_course_tab)
        self.view_courses()
        self.notebook.add(self.add_course_tab, text="Courses")
        self.notebook.pack(fill=BOTH, expand=1)

        # Add teacher tab
        self.add_teacher_tab = Frame(self.notebook, bg="green")
        self.create_left_icon_teachers(self.add_teacher_tab)
        self.create_label_frame_teachers(self.add_teacher_tab)
        self.create_message_area_teachers(self.add_teacher_tab)
        self.create_tree_view_teacher()
        self.create_scrollbar_teachers(self.add_teacher_tab, self.tree_teachers)
        self.create_bottom_buttons_teachers(self.add_teacher_tab)
        self.view_teachers()
        self.notebook.add(self.add_teacher_tab, text="Teachers")
        self.notebook.pack(fill=BOTH, expand=1)

        # Add student tab
        self.add_student_tab = Frame(self.notebook, bg="yellow")
        self.create_left_icon_students(self.add_student_tab)
        self.create_label_frame_students(self.add_student_tab)
        self.create_message_area_students(self.add_student_tab)
        self.create_tree_view_student()
        self.create_scrollbar_students(self.add_student_tab, self.tree_students)
        self.create_bottom_buttons_students(self.add_student_tab)
        self.view_students()
        self.notebook.add(self.add_student_tab, text="Students")
        self.notebook.pack(fill=BOTH, expand=1)

    def create_left_icon_courses(self, tab):
        photo_courses = PhotoImage(file="logo_courses.gif")
        self.label_courses = Label(tab, image=photo_courses)
        self.label_courses.image = photo_courses
        self.label_courses.grid(row=0, column=0, padx=20, pady=20)

    def create_left_icon_teachers(self, tab):
        photo_teachers = PhotoImage(file="logo_teachers.gif")
        self.label_teachers = Label(tab, image=photo_teachers)
        self.label_teachers.image = photo_teachers
        self.label_teachers.grid(row=0, column=0, padx=20, pady=20)

    def create_left_icon_students(self, tab):
        photo_students = PhotoImage(file="logo_students.gif")
        self.label_students = Label(tab, image=photo_students)
        self.label_students.image = photo_students
        self.label_students.grid(row=0, column=0, padx=20, pady=20)

    def create_label_frame_courses(self, tab):
        self.course_labelframe = LabelFrame(tab, text="Create New Course", bg="#42bcf5", font="Helvetica 10")
        self.course_labelframe.grid(row=0, column=1, padx=8, pady=20)

        self.course_name_labelfield = Label(self.course_labelframe, text="Name: ", bg="sky blue", fg="black")
        self.course_name_labelfield.grid(row=1, column=1, sticky="W", padx=2, pady=8)
        self.course_name_field = Entry(self.course_labelframe, width=40)
        self.course_name_field.grid(row=1, column=2, padx=15, pady=1)

        self.course_teacher_labelfield = Label(self.course_labelframe, text="Teacher: ", bg="sky blue", fg="black")
        self.course_teacher_labelfield.grid(row=3, column=1, sticky="W", padx=2, pady=8)
        self.course_teacher_field = Entry(self.course_labelframe, width=40)
        self.course_teacher_field.grid(row=3, column=2, padx=15, pady=10)

        self.course_credits_labelfield = Label(self.course_labelframe, text="Credits: ", bg="sky blue", fg="black")
        self.course_credits_labelfield.grid(row=5, column=1, sticky="W", padx=2, pady=8)
        self.course_credits_field = Entry(self.course_labelframe, width=40)
        self.course_credits_field.grid(row=5, column=2, padx=15, pady=10)

        self.course_add_button = Button(self.course_labelframe, text="Add Course", bg="blue", fg="white", width=50,
                                        command=self.on_add_course_button_clicked)
        self.course_add_button.grid(row=6, column=1, columnspan=2, pady=5)

    def create_label_frame_teachers(self, tab):
        self.teacher_labelframe = LabelFrame(tab, text="Add New Teacher", bg="green", font="Helvetica 10")
        self.teacher_labelframe.grid(row=0, column=1, padx=8, pady=20)

        self.teacher_firstname_labelfield = Label(self.teacher_labelframe, text="First Name: ", bg="sea green",
                                                  fg="black")
        self.teacher_firstname_labelfield.grid(row=1, column=1, sticky="W", padx=2, pady=8)
        self.teacher_firstname_field = Entry(self.teacher_labelframe, width=40)
        self.teacher_firstname_field.grid(row=1, column=2, padx=15, pady=1)

        self.teacher_lastname_labelfield = Label(self.teacher_labelframe, text="Last Name: ", bg="sea green",
                                                 fg="black")
        self.teacher_lastname_labelfield.grid(row=3, column=1, sticky="W", padx=2, pady=8)
        self.teacher_lastname_field = Entry(self.teacher_labelframe, width=40)
        self.teacher_lastname_field.grid(row=3, column=2, padx=15, pady=10)

        self.teacher_email_labelfield = Label(self.teacher_labelframe, text="Email: ", bg="sea green", fg="black")
        self.teacher_email_labelfield.grid(row=5, column=1, sticky="W", padx=2, pady=8)
        self.teacher_email_field = Entry(self.teacher_labelframe, width=40)
        self.teacher_email_field.grid(row=5, column=2, padx=15, pady=10)

        self.teacher_number_labelfield = Label(self.teacher_labelframe, text="Phone Number: ", bg="sea green",
                                               fg="black")
        self.teacher_number_labelfield.grid(row=7, column=1, sticky="W", padx=2, pady=8)
        self.teacher_number_field = Entry(self.teacher_labelframe, width=40)
        self.teacher_number_field.grid(row=7, column=2, padx=15, pady=10)

        self.teacher_add_button = Button(self.teacher_labelframe, text="Add Teacher", bg="sea green", fg="white",
                                         width=50, command=self.on_add_teacher_button_clicked)
        self.teacher_add_button.grid(row=8, column=1, columnspan=2, pady=5)

    def create_label_frame_students(self, tab):
        self.student_labelframe = LabelFrame(tab, text="Add New Student", bg="yellow", font="Helvetica 10")
        self.student_labelframe.grid(row=0, column=1, padx=8, pady=20)

        self.student_firstname_labelfield = Label(self.student_labelframe, text="First Name: ", bg="gold", fg="black")
        self.student_firstname_labelfield.grid(row=1, column=1, sticky="W", padx=2, pady=8)
        self.student_firstname_field = Entry(self.student_labelframe, width=40)
        self.student_firstname_field.grid(row=1, column=2, padx=15, pady=1)

        self.student_lastname_labelfield = Label(self.student_labelframe, text="Last Name: ", bg="gold", fg="black")
        self.student_lastname_labelfield.grid(row=3, column=1, sticky="W", padx=2, pady=8)
        self.student_lastname_field = Entry(self.student_labelframe, width=40)
        self.student_lastname_field.grid(row=3, column=2, padx=15, pady=1)

        self.student_email_labelfield = Label(self.student_labelframe, text="Email: ", bg="gold", fg="black")
        self.student_email_labelfield.grid(row=5, column=1, sticky="W", padx=2, pady=8)
        self.student_email_field = Entry(self.student_labelframe, width=40)
        self.student_email_field.grid(row=5, column=2, padx=15, pady=10)

        self.student_number_labelfield = Label(self.student_labelframe, text="Phone Number: ", bg="gold", fg="black")
        self.student_number_labelfield.grid(row=7, column=1, sticky="W", padx=2, pady=8)
        self.student_number_field = Entry(self.student_labelframe, width=40)
        self.student_number_field.grid(row=7, column=2, padx=15, pady=10)

        self.student_add_button = Button(self.student_labelframe, text="Add Student", bg="gold", fg="white", width=50,
                                         command=self.on_add_student_button_clicked)
        self.student_add_button.grid(row=8, column=1, columnspan=2, pady=5)

    def create_message_area_courses(self, tab):
        self.courses_message = Label(tab, text="", bg="#42bcf5", fg="black")
        self.courses_message.grid(row=3, column=1)

    def create_message_area_teachers(self, tab):
        self.teachers_message = Label(tab, text="", bg="green", fg="black")
        self.teachers_message.grid(row=3, column=1)

    def create_message_area_students(self, tab):
        self.students_message = Label(tab, text="", bg="yellow", fg="black")
        self.students_message.grid(row=3, column=1)

    def create_tree_view_course(self):
        self.tree_courses = ttk.Treeview(self.add_course_tab, height=10, columns=("teacher", "credits"))
        self.tree_courses.grid(row=6, column=0, columnspan=3)
        self.tree_courses.heading("#0", text="Name", anchor=W)
        self.tree_courses.heading("teacher", text="Teacher", anchor=W)
        self.tree_courses.heading("credits", text="Credits", anchor=W)

    def create_tree_view_teacher(self):
        self.tree_teachers = ttk.Treeview(self.add_teacher_tab, height=10, columns=("last_name", "email", "number"))
        self.tree_teachers.grid(row=6, column=0, columnspan=3)
        self.tree_teachers.heading("#0", text="First Name", anchor=W)
        self.tree_teachers.heading("last_name", text="Last Name", anchor=W)
        self.tree_teachers.heading("email", text="Email Address", anchor=W)
        self.tree_teachers.heading("number", text="Contact Number", anchor=W)

    def create_tree_view_student(self):
        self.tree_students = ttk.Treeview(self.add_student_tab, height=10, columns=("last_name", "email", "number"))
        self.tree_students.grid(row=6, column=0, columnspan=3)
        self.tree_students.heading("#0", text="First Name", anchor=W)
        self.tree_students.heading("last_name", text="Last Name", anchor=W)
        self.tree_students.heading("email", text="Email Address", anchor=W)
        self.tree_students.heading("number", text="Contact Number", anchor=W)

    def create_scrollbar_courses(self, tab, tree):
        self.courses_scrollbar = Scrollbar(tab, orient="vertical", command=tree.yview)
        self.courses_scrollbar.grid(row=6, column=3, columnspan=10, sticky="sn")

    def create_scrollbar_teachers(self, tab, tree):
        self.teachers_scrollbar = Scrollbar(tab, orient="vertical", command=tree.yview)
        self.teachers_scrollbar.grid(row=6, column=3, columnspan=10, sticky="sn")

    def create_scrollbar_students(self, tab, tree):
        self.students_scrollbar = Scrollbar(tab, orient="vertical", command=tree.yview)
        self.students_scrollbar.grid(row=6, column=3, columnspan=10, sticky="sn")

    def create_bottom_buttons_courses(self, tab):
        self.delete_button_courses = Button(tab, text="Delete Selected",
                                            command=lambda: self.on_delete_course_selected_button_clicked(),
                                            bg="sky blue", fg="black")
        self.delete_button_courses.grid(row=8, column=0, sticky=W, padx=20, pady=10)

        self.modify_button_courses = Button(tab, text="Modify Selected",
                                            command=lambda: self.on_update_course_selected_button_clicked(),
                                            bg="purple", fg="white")
        self.modify_button_courses.grid(row=8, column=1, sticky=E, padx=20, pady=10)

    def create_bottom_buttons_teachers(self, tab):
        self.delete_button_teachers = Button(tab, text="Delete Selected",
                                             command=lambda: self.on_delete_teacher_selected_button_clicked(),
                                             bg="sea green", fg="black")
        self.delete_button_teachers.grid(row=8, column=0, sticky=W, padx=20, pady=10)

        self.modify_button_teachers = Button(tab, text="Modify Selected",
                                             command=lambda: self.on_update_teacher_selected_button_clicked(),
                                             bg="purple", fg="white")
        self.modify_button_teachers.grid(row=8, column=1, sticky=E, padx=20, pady=10)

    def create_bottom_buttons_students(self, tab):
        self.delete_button_students = Button(tab, text="Delete Selected",
                                             command=lambda: self.on_delete_student_selected_button_clicked(),
                                             bg="gold", fg="black")
        self.delete_button_students.grid(row=8, column=0, sticky=W, padx=20, pady=10)

        self.modify_button_students = Button(tab, text="Modify Selected",
                                             command=lambda: self.on_update_student_selected_button_clicked(),
                                             bg="purple", fg="white")
        self.modify_button_students.grid(row=8, column=1, sticky=E, padx=20, pady=10)

    def view_courses(self):
        items = self.tree_courses.get_children()
        for item in items:
            self.tree_courses.delete(item)
        query = "SELECT C.NAME, T.FIRST_NAME || ' ' || T.LAST_NAME, C.CREDITS FROM COURSES C LEFT JOIN TEACHERS T " \
                "ON C.TEACHER_ID = T.ID ORDER BY C.NAME DESC"
        contact_entries = self.data_query.execute_db_query(self.back, query)
        for row in contact_entries:
            self.tree_courses.insert("", 0, text=row[0], values=(row[1], row[2]))

    def view_teachers(self):
        items = self.tree_teachers.get_children()
        for item in items:
            self.tree_teachers.delete(item)
        query = "SELECT FIRST_NAME, LAST_NAME, EMAIL, NUMBER FROM TEACHERS ORDER BY FIRST_NAME DESC, LAST_NAME DESC"
        contact_entries = self.data_query.execute_db_query(self.back, query)
        for row in contact_entries:
            self.tree_teachers.insert("", 0, text=row[0], values=(row[1], row[2], row[3]))

    def view_students(self):
        items = self.tree_students.get_children()
        for item in items:
            self.tree_students.delete(item)
        query = "SELECT FIRST_NAME, LAST_NAME, EMAIL, NUMBER FROM STUDENTS ORDER BY FIRST_NAME DESC, LAST_NAME DESC"
        contact_entries = self.data_query.execute_db_query(self.back, query)
        for row in contact_entries:
            self.tree_students.insert("", 0, text=row[0], values=(row[1], row[2], row[3]))

    def on_add_course_button_clicked(self):
        self.add_new_course()

    def on_add_teacher_button_clicked(self):
        self.add_new_teacher()

    def on_add_student_button_clicked(self):
        self.add_new_student()

    def new_course_validated(self):
        try:
            if len(self.course_name_field.get()) <= 1:
                self.courses_message["text"] = "Course name must have more than one character."
                return False
            if self.course_name_field.get().isnumeric():
                self.courses_message["text"] = "Course name can't be a number."
                return False
            if len(self.course_teacher_field.get()) <= 1:
                self.courses_message["text"] = "Teacher's name must have more than one character."
                return False
            teacher_name = self.course_teacher_field.get().split(" ")
            if len(teacher_name) != 2:
                self.courses_message["text"] = "Teacher must have first name and last name!"
                return False
            if not self.course_credits_field.get().isnumeric():
                self.courses_message["text"] = "Credits must be a positive number!"
                return False
            if int(self.course_credits_field.get()) <= 0:
                self.courses_message["text"] = "Credits must be positive!"
                return False
            query = "SELECT ID FROM COURSES WHERE LOWER(NAME) = LOWER(TRIM(?))"
            id_course = self.back.select(query, (self.course_name_field.get(),))
            if id_course:
                self.courses_message["text"] = "Course already exists!"
                return False
            return True
        except Exception as e:
            self.courses_message["text"] = "ERROR! " + str(e)
            return False

    def new_teacher_validated(self):
        regex = "^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$"
        try:
            if len(self.teacher_firstname_field.get()) <= 1:
                self.teachers_message["text"] = "Teacher's first name must have more than one character."
                return False
            if self.teacher_firstname_field.get().isnumeric():
                self.teachers_message["text"] = "Teacher's first name can't be a number."
                return False
            if len(self.teacher_lastname_field.get()) <= 1:
                self.teachers_message["text"] = "Teacher's last name must have more than one character."
                return False
            if self.teacher_lastname_field.get().isnumeric():
                self.teachers_message["text"] = "Teacher's last name can't be a number."
                return False
            query = "SELECT ID FROM TEACHERS WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                    "AND LOWER(LAST_NAME)= LOWER(TRIM(?))"
            id_teacher = self.back.select(query, (self.teacher_firstname_field.get(),
                                                  self.teacher_lastname_field.get()))
            if id_teacher:
                self.teachers_message["text"] = "Teacher already registered!"
                return False
            if re.search(regex, self.teacher_email_field.get().lower()) is None:
                self.teachers_message["text"] = "Invalid email address."
                return False
            if self.teacher_number_field.get() == "":
                self.teachers_message["text"] = "Number must be filled."
                return False
            if not self.teacher_number_field.get().isnumeric():
                self.teachers_message["text"] = "Phone Number must contain only numbers."
                return False
            if int(self.teacher_number_field.get()) < 0 or len(self.teacher_number_field.get()) < 10:
                self.teachers_message["text"] = "Number must be positive and have 10 characters."
                return False
            return True
        except Exception as e:
            self.teachers_message["text"] = "ERROR! " + str(e)
            return False

    def new_student_validated(self):
        try:
            regex = "^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$"
            if len(self.student_firstname_field.get()) <= 1:
                self.students_message["text"] = "Student's first name must have more than one character."
                return False
            if self.student_firstname_field.get().isnumeric():
                self.students_message["text"] = "Student's first name can't be a number."
                return False
            if len(self.student_lastname_field.get()) <= 1:
                self.students_message["text"] = "Student's last name must have more than one character."
                return False
            if self.student_lastname_field.get().isnumeric():
                self.students_message["text"] = "Student's last name can't be a number."
                return False
            query = "SELECT ID FROM STUDENTS WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                    "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
            id_student = self.back.select(query, (self.student_firstname_field.get(),
                                                  self.student_lastname_field.get()))
            if id_student:
                self.students_message["text"] = "Student already registered!"
                return False
            if re.search(regex, self.student_email_field.get().lower()) is None:
                self.students_message["text"] = "Invalid email address."
                return False
            if self.student_number_field.get() == "":
                self.students_message["text"] = "Number must be filled."
                return False
            if not self.student_number_field.get().isnumeric():
                self.students_message["text"] = "Phone Number must contain only numbers."
                return False
            if int(self.student_number_field.get()) < 0 or len(self.student_number_field.get()) < 10:
                self.students_message["text"] = "Number must be positive and have 10 characters."
                return False
            return True
        except Exception as e:
            self.students_message["text"] = "ERROR! " + str(e)
            return False

    def validate_teacher(self, name):
        query = "SELECT ID FROM TEACHERS WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
        teacher_name = name.split(" ")
        if len(teacher_name) != 2:
            return "Teacher must have first name and last name!"
        id_teacher = self.back.select(query, (teacher_name[0], teacher_name[1]))
        if not id_teacher:
            return "Teacher not registered!"
        return id_teacher[0][0]

    def add_new_course(self):
        if self.new_course_validated():
            msg_teacher = self.validate_teacher(self.course_teacher_field.get())
            if msg_teacher != "":
                self.courses_message["text"] = msg_teacher

            query = "SELECT ID FROM TEACHERS WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                    "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
            teacher_name = self.course_teacher_field.get().split(" ")
            if len(teacher_name) != 2:
                self.courses_message["text"] = "Teacher not registered!"
                return
            id_teacher = self.back.select(query, (teacher_name[0].capitalize(), teacher_name[1].capitalize()))
            if not id_teacher:
                self.courses_message["text"] = "Teacher not registered!"
                return
            query = "INSERT INTO COURSES (NAME, TEACHER_ID, CREDITS) VALUES (TRIM(?), ?, ?)"
            parameters = (self.course_name_field.get().capitalize(), id_teacher[0][0], self.course_credits_field.get())
            self.data_query.execute_db_query(self.back, query, parameters)
            self.courses_message["text"] = "New course {} added.".format(self.course_name_field.get().capitalize())
            self.course_name_field.delete(0, END)
            self.course_teacher_field.delete(0, END)
            self.course_credits_field.delete(0, END)
            self.view_courses()

    def add_new_teacher(self):
        if self.new_teacher_validated():
            query = "INSERT INTO TEACHERS (FIRST_NAME, LAST_NAME, EMAIL, NUMBER) VALUES (TRIM(?), TRIM(?), " \
                    "TRIM(?), ?)"
            parameters = (self.teacher_firstname_field.get().capitalize(),
                          self.teacher_lastname_field.get().capitalize(),
                          self.teacher_email_field.get().lower(), self.teacher_number_field.get())
            self.data_query.execute_db_query(self.back, query, parameters)
            self.teachers_message["text"] = "New teacher {} {} added.".format(
                self.teacher_firstname_field.get().capitalize(), self.teacher_lastname_field.get().capitalize())
            self.teacher_firstname_field.delete(0, END)
            self.teacher_lastname_field.delete(0, END)
            self.teacher_email_field.delete(0, END)
            self.teacher_number_field.delete(0, END)
            self.view_teachers()

    def add_new_student(self):
        if self.new_student_validated():
            query = "INSERT INTO STUDENTS (FIRST_NAME, LAST_NAME, EMAIL, NUMBER) VALUES (TRIM(?), TRIM(?), " \
                    "TRIM(?), ?)"
            parameters = (self.student_firstname_field.get().capitalize(),
                          self.student_lastname_field.get().capitalize(), self.student_email_field.get().lower(),
                          self.student_number_field.get())
            self.data_query.execute_db_query(self.back, query, parameters)
            self.students_message["text"] = "New student {} {} added.".format(
                self.student_firstname_field.get().capitalize(), self.student_lastname_field.get().capitalize())
            self.student_firstname_field.delete(0, END)
            self.student_lastname_field.delete(0, END)
            self.student_email_field.delete(0, END)
            self.student_number_field.delete(0, END)
            self.view_students()

    def on_delete_course_selected_button_clicked(self):
        self.courses_message["text"] = ""
        try:
            if len(self.tree_courses.selection()) > 1:
                self.courses_message["text"] = "You must select only one course!"
                return
            self.tree_courses.item(self.tree_courses.selection())["values"][0]
        except IndexError:
            self.courses_message["text"] = "No course selected to delete!"
            return
        self.delete_course()

    def on_delete_teacher_selected_button_clicked(self):
        self.teachers_message["text"] = ""
        try:
            if len(self.tree_teachers.selection()) > 1:
                self.teachers_message["text"] = "You must select only one teacher!"
                return
            self.tree_teachers.item(self.tree_teachers.selection())["values"][0]
        except IndexError:
            self.teachers_message["text"] = "No teacher selected to delete!"
            return
        self.delete_teacher()

    def on_delete_student_selected_button_clicked(self):
        self.students_message["text"] = ""
        try:
            if len(self.tree_students.selection()) > 1:
                self.students_message["text"] = "You must select only one student!"
                return
            self.tree_students.item(self.tree_students.selection())["values"][0]
        except IndexError:
            self.students_message["text"] = "No student selected to delete!"
            return
        self.delete_student()

    def delete_course(self):
        self.courses_message["text"] = ""
        name = self.tree_courses.item(self.tree_courses.selection())["text"]
        query = "DELETE FROM COURSES WHERE LOWER(NAME) = LOWER(TRIM(?))"
        self.data_query.execute_db_query(self.back, query, (name,))
        self.courses_message["text"] = f"Course {name} deleted."
        self.view_courses()

    def delete_teacher(self):
        self.teachers_message["text"] = ""
        first_name = self.tree_teachers.item(self.tree_teachers.selection())["text"]
        last_name = self.tree_teachers.item(self.tree_teachers.selection())["values"][0]

        query = "SELECT ID FROM TEACHERS WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
        teacher_id = self.back.select(query, (first_name, last_name))

        query = "DELETE FROM TEACHERS WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
        self.data_query.execute_db_query(self.back, query, (first_name, last_name))
        self.teachers_message["text"] = f"Teacher {first_name} {last_name} deleted."
        self.view_teachers()

        # Clear teacher from all courses is assigned to
        query = "UPDATE COURSES SET TEACHER_ID = 0 WHERE TEACHER_ID = ?"
        self.data_query.execute_db_query(self.back, query, (teacher_id[0][0],))
        self.view_courses()

    def delete_student(self):
        self.students_message["text"] = ""
        first_name = self.tree_students.item(self.tree_students.selection())["text"]
        last_name = self.tree_students.item(self.tree_students.selection())["values"][0]
        query = "DELETE FROM STUDENTS WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
        self.data_query.execute_db_query(self.back, query, (first_name, last_name))
        self.students_message["text"] = f"Student {first_name} {last_name} deleted."
        self.view_students()

    def on_update_course_selected_button_clicked(self):
        self.courses_message["text"] = ""
        try:
            if len(self.tree_courses.selection()) > 1:
                self.courses_message["text"] = "You must select only one course!"
                return
            self.tree_courses.item(self.tree_courses.selection())["values"][0]
        except IndexError:
            self.courses_message["text"] = "No course selected to update!"
            return
        self.open_modify_window_courses()

    def on_update_teacher_selected_button_clicked(self):
        self.teachers_message["text"] = ""
        try:
            if len(self.tree_teachers.selection()) > 1:
                self.teachers_message["text"] = "You must select only one teacher!"
                return
            self.tree_teachers.item(self.tree_teachers.selection())["values"][0]
        except IndexError:
            self.teachers_message["text"] = "No teacher selected to update!"
            return
        self.open_modify_window_teachers()

    def on_update_student_selected_button_clicked(self):
        self.students_message["text"] = ""
        try:
            if len(self.tree_students.selection()) > 1:
                self.students_message["text"] = "You must select only one student!"
                return
            self.tree_students.item(self.tree_students.selection())["values"][0]
        except IndexError:
            self.students_message["text"] = "No student selected to update!"
            return
        self.open_modify_window_students()

    def open_modify_window_courses(self):
        name = self.tree_courses.item(self.tree_courses.selection())["text"]
        old_teacher = self.tree_courses.item(self.tree_courses.selection())["values"][0]
        self.transient = Toplevel()
        self.transient.title("Update Course")
        Label(self.transient, text="Name: ").grid(row=0, column=1)
        Entry(self.transient, textvariable=StringVar(self.transient, value=name),
              state="readonly").grid(row=0, column=2)

        Label(self.transient, text="Old Teacher: ").grid(row=1, column=1)
        Entry(self.transient, textvariable=StringVar(self.transient, value=old_teacher),
              state="readonly").grid(row=1, column=2)

        Label(self.transient, text="New Teacher: ").grid(row=2, column=1)
        update_course_widget = Entry(self.transient)
        update_course_widget.grid(row=2, column=2)

        Button(self.transient, text="Update Course",
               command=lambda: self.update_course(update_course_widget.get(), name)).grid(row=3, column=2, sticky=E)

    def open_modify_window_teachers(self):
        first_name = self.tree_teachers.item(self.tree_teachers.selection())["text"]
        last_name = self.tree_teachers.item(self.tree_teachers.selection())["values"][0]
        old_number = self.tree_teachers.item(self.tree_teachers.selection())["values"][2]
        self.transient = Toplevel()
        self.transient.title("Update Teacher")
        Label(self.transient, text="Name: ").grid(row=0, column=1)
        Entry(self.transient, textvariable=StringVar(self.transient, value=first_name + " " + last_name),
              state="readonly").grid(row=0, column=2)

        Label(self.transient, text="Old Phone Number: ").grid(row=1, column=1)
        Entry(self.transient, textvariable=StringVar(self.transient, value=old_number),
              state="readonly").grid(row=1, column=2)

        Label(self.transient, text="New Phone Number: ").grid(row=2, column=1)
        update_teacher_number_widget = Entry(self.transient)
        update_teacher_number_widget.grid(row=2, column=2)

        Button(self.transient, text="Update Teacher",
               command=lambda: self.update_teacher(update_teacher_number_widget.get(), first_name,
                                                   last_name)).grid(row=3, column=2, sticky=E)

    def open_modify_window_students(self):
        first_name = self.tree_students.item(self.tree_students.selection())["text"]
        last_name = self.tree_students.item(self.tree_students.selection())["values"][0]
        old_number = self.tree_students.item(self.tree_students.selection())["values"][2]
        self.transient = Toplevel()
        self.transient.title("Update Student")
        Label(self.transient, text="Name: ").grid(row=0, column=1)
        Entry(self.transient, textvariable=StringVar(self.transient, value=first_name + " " + last_name),
              state="readonly").grid(row=0, column=2)

        Label(self.transient, text="Old Phone Number: ").grid(row=1, column=1)
        Entry(self.transient, textvariable=StringVar(self.transient, value=old_number),
              state="readonly").grid(row=1, column=2)

        Label(self.transient, text="New Phone Number: ").grid(row=2, column=1)
        update_student_number_widget = Entry(self.transient)
        update_student_number_widget.grid(row=2, column=2)

        Button(self.transient, text="Update Student",
               command=lambda: self.update_student(update_student_number_widget.get(), first_name,
                                                   last_name)).grid(row=3, column=2, sticky=E)

    def update_course(self, new_teacher, course_name):
        # validate_teacher returns the id if exists, otherwise a error message
        new_teacher_id = self.validate_teacher(new_teacher)
        if new_teacher == "":
            self.courses_message["text"] = "New teacher must be filled."
        elif isinstance(new_teacher_id, int):
            query = "SELECT ID FROM COURSES WHERE LOWER(NAME) = LOWER(TRIM(?))"
            course_id = self.back.select(query, (course_name,))
            course_id = course_id[0][0]

            query = "UPDATE COURSES SET TEACHER_ID = ? WHERE ID = ?"
            parameters = (new_teacher_id, course_id)
            ret = self.data_query.execute_db_query(self.back, query, parameters)
            self.transient.destroy()
            self.courses_message["text"] = "Teacher of {} modified to {}.".format(course_name, new_teacher.title())
        else:
            self.courses_message["text"] = new_teacher_id
        self.view_courses()

    def update_teacher(self, new_number, first_name, last_name):
        if new_number == "":
            self.teachers_message["text"] = "New number must be filled."
        elif not new_number.isnumeric():
            self.teachers_message["text"] = "Number must contain only numbers."
        elif int(new_number) < 0 or len(new_number) < 10:
            self.teachers_message["text"] = "Number invalid."
        else:
            query = "UPDATE TEACHERS SET NUMBER = ? WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                    "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
            parameters = (new_number, first_name, last_name)
            self.data_query.execute_db_query(self.back, query, parameters)
            self.transient.destroy()
            self.teachers_message["text"] = "Phone number of {} modified.".format(first_name.title() + " " +
                                                                                  last_name.title())
            self.view_teachers()

    def update_student(self, new_number, first_name, last_name):
        if new_number == "":
            self.students_message["text"] = "New number must be filled."
        elif not new_number.isnumeric():
            self.students_message["text"] = "Number must contain only numbers."
        elif int(new_number) < 0 or len(new_number) < 10:
            self.teachers_message["text"] = "Number invalid."
        else:
            query = "UPDATE STUDENTS SET NUMBER = ? WHERE LOWER(FIRST_NAME) = LOWER(TRIM(?)) " \
                    "AND LOWER(LAST_NAME) = LOWER(TRIM(?))"
            parameters = (new_number, first_name, last_name)
            self.data_query.execute_db_query(self.back, query, parameters)
            self.transient.destroy()
            self.students_message["text"] = "Phone number of {} modified.".format(first_name.title() + " " +
                                                                                  last_name.title())
            self.view_students()

    def on_close_window(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            exit(0)
