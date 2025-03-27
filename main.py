
from pushbullet import Pushbullet
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import tensorflow as tf
import cv2
import numpy as np
import time 
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from playsound import playsound
import time
import os
import sys
import shutil
import tkinter as tk
API_key = "" # API key for pushbullet.
text = ""

pb = Pushbullet(API_key)
log_messages= []

class InputGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Input Form")
        
        large_font = ("Arial", 12)

        # Labels and Entry widgets
        tk.Label(root, text="email:", font= large_font).grid(row=0, column=0, padx=10, pady=5)
        self.email_entry = tk.Entry(root, font=large_font, width=25)
        self.email_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(root, text="Password:", font= large_font).grid(row=1, column=0, padx=10, pady=5)
        self.password_entry = tk.Entry(root, font=large_font, width=25)
        self.password_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(root, text="First name:", font= large_font).grid(row=2, column=0, padx=10, pady=5)
        self.first_name_entry = tk.Entry(root, font=large_font, width=25)
        self.first_name_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(root, text="Last name:", font= large_font).grid(row=3, column=0, padx=10, pady=5)
        self.last_name_entry = tk.Entry(root, font=large_font, width=25)
        self.last_name_entry.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(root, text="Row:", font= large_font).grid(row=4, column=0, padx=10, pady=5)
        self.row_entry = tk.Entry(root, font=large_font, width=25)
        self.row_entry.grid(row=4, column=1, padx=10, pady=5)

        tk.Label(root, text="Column:", font= large_font).grid(row=5, column=0, padx=10, pady=5)
        self.col_entry = tk.Entry(root, font=large_font, width=25)
        self.col_entry.grid(row=5, column=1, padx=10, pady=5)

        tk.Label(root, text="Center (3 letters):", font= large_font).grid(row=6, column=0, padx=10, pady=5)
        self.center_entry = tk.Entry(root, font=large_font, width=25)
        self.center_entry.grid(row=6, column=1, padx=10, pady=5)

        tk.Label(root, text="Null Osta num:", font= large_font).grid(row=7, column=0, padx=10, pady=5)
        self.null_osta_entry = tk.Entry(root, font=large_font, width=25)
        self.null_osta_entry.grid(row=7, column=1, padx=10, pady=5)

        tk.Label(root, text="cap num:", font= large_font).grid(row=8, column=0, padx=10, pady=5)
        self.cap_num = tk.Entry(root, font=large_font, width=25)
        self.cap_num.grid(row=8, column=1, padx=10, pady=5)

        tk.Label(root, text="key:", font= large_font).grid(row=9, column=0, padx=10, pady=5)
        self.key = tk.Entry(root, font=large_font, width=25)
        self.key.grid(row=9, column=1, padx=10, pady=5)

        # Submit button
        submit_button = tk.Button(root, text="Submit", font= large_font, command=self.show)
        submit_button.grid(row=10, columnspan=2, pady=10)
        
    def show(self):
        self.email = self.email_entry.get()
        self.password = self.password_entry.get()
        self.first_name = self.first_name_entry.get()
        self.last_name = self.last_name_entry.get()
        self.row = self.row_entry.get()
        self.col = self.col_entry.get()
        self.center = self.center_entry.get()
        self.null_osta = self.null_osta_entry.get()
        self.cap_num = self.cap_num.get()
        self.key = self.key.get()
        # Add your validation here if necessary
        if self.row and self.col and self.email and self.password and self.first_name and self.last_name:
            self.root.quit()  # Close the window
            self.root.destroy()
        else:
            tk.messagebox.showerror("Input Error", "Please fill out all fields")

    def run(self):
        self.root.mainloop()
        return self.email, self.password, self.first_name, self.last_name, self.row, self.col, self.center, self.null_osta, self.cap_num, self.key

gui = InputGUI(tk.Tk())
email, password, first_name, last_name, row, col, center, null_osta, cap_num, key = gui.run()
row = int(row)
col = int(col)


if getattr(sys, 'frozen', False):
    # Running as compiled .exe
    base_path = os.path.dirname(sys.executable)
else:
    # Running as script
    base_path = os.path.dirname(os.path.abspath(__file__))

dir = os.path.join(base_path, "log1")

model_path = os.path.join(base_path, "quantized_model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()



def pred_cap(cap, image_height, image_width, image_channels):
    image = cv2.imread(cap)
    red_channel = image[:, :, 2]
    ret, clean_image = cv2.threshold(red_channel, 80, 255, cv2.THRESH_BINARY_INV)
    contours, heirarchy = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    preprocessed_digits = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        #adding or subtracting pixels to get the complete digit in the rectangle
        cv2.rectangle(clean_image, (x - 3,y - 3), (x+w+ 2, y+h + 2), color=(0, 255, 0), thickness=2)
        
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = clean_image[y:y+h, x:x+w]
        
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))
        resized_digit = resized_digit.astype(np.float32)  # Convert to float32
        resized_digit = resized_digit / 255.0
        
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        
        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)

    preprocessed_digits = np.array(preprocessed_digits)
    preprocessed_digits = preprocessed_digits.reshape((len(preprocessed_digits), image_height, image_width, image_channels))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    cap_text = ''
    for digit in preprocessed_digits:
        # Reshape the input to match the model's expected input shape
        input_data = np.expand_dims(digit, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted digit
        predicted_digit = np.argmax(output_data)
        cap_text += str(predicted_digit)
    
    return cap_text


def solve_cap(driver):
    start_time = time.time()
    js_script = """
    var img = document.querySelector('img#Imageid');
    return img ? img.src : null;
    """

    image_url = driver.execute_script(js_script)
    print("cap 1")
    response = requests.get(image_url)

    # Check if the request was successful
    
        # Save the image locally
    with open('image.jpg', 'wb') as f:
        f.write(response.content)
        print("Image downloaded successfully.")
   
    prediction = pred_cap("image.jpg", image_height= 28, image_width= 28, image_channels= 1)
    del response
    check = """
    var element = document.getElementById('captcha_code_reg');
    return element ? element : 'Element not found';
    """

    # Execute JavaScript and get the result
    driver.execute_script(check)
    text_sending = """
    var element = document.querySelector('#captcha_code_reg');
    if (element) {
        element.value = arguments[0];
        
        // Create and dispatch an 'input' event
        var event = new Event('input', { bubbles: true });
        element.dispatchEvent(event);
        
        // Create and dispatch a 'change' event
        var changeEvent = new Event('change', { bubbles: true });
        element.dispatchEvent(changeEvent);
        
        return true;
    }
    return false;
    """
    driver.execute_script(text_sending, prediction)
    end_time = time.time()
    print(f"prediction time: {end_time - start_time} seconds")


def login(driver):
    start_time = time.time()
    if (driver.current_url == "https://blsitalypakistan.com/"):
        driver.get("https://blsitalypakistan.com/account/login")
        WebDriverWait(driver, 3).until(
            lambda d: d.current_url == "https://blsitalypakistan.com/account/login"
        )
    if (driver.current_url == "https://blsitalypakistan.com/account/login"):
        try:
        # Wait until the close_button with the specified class name is visible or 10 seconds have passed
            close_button = WebDriverWait(driver, 0.1).until(
                EC.element_to_be_clickable((By.XPATH, "//a[@class='cl']"))
            )

        # Click the close button
            close_button.click()
            print("pop up is visible")
        except:
            print("e")
        try:
            if cap_num == 1:
                solve_cap(driver)
            else:
                print("no c.")

            email_field = WebDriverWait(driver, 3).until(
                EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Enter Email']"))
            )
            email_field.send_keys(email)
            pass_field = driver.find_element(By.XPATH, "//input[@placeholder='Enter Password']")
            pass_field.send_keys(password)
            
            driver.find_element(By.XPATH, "//button[@name='submitLogin']").click()
            end_time = time.time()
            print(f"Login method time: {end_time - start_time} seconds")
        
        except Exception as e:
            print(e)
            print("cant login")
            driver.get("https://blsitalypakistan.com/account/login")
            login(driver)
    else:
        print("already logged in")


def check_for_appointment(driver):
    isl_study = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/MUpTVkt3ODA5OTU2MzE4NjQ/NjltcVJRUzcyODcyODU5NTkz/MXdOc0tMOTMyNzM1MDY2MTg"
    Quetta = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/NkZMdW50Mzc0MTc5MjgwNjY/ODdzZ2VySTU4MjY3MDcxNDUx/MXdGVVpBMzAwNTI0MzE5NzQ"
    Family_fsl = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/NHVRTWhqODQxNDAyOTc1Njk/ODNuTE9pajIyNDcxMTc2ODM2/MkZZYUpYMjM3MTU2NDUxMjc"
    Lahore_study = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/MlBzSVhjMDQzNDg2NTcwOTM/NzlESlhubTAwNDYzNDgxMTc5/MXl6b2FSNDI4MTEzOTA0NTc"
    quetta_fml = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/NnVSbVRuOTQwNzY1NjkwMTM/ODhYQkV6TTIzNTg0MTA5MTc3/MUpPc1NaMzA1Nzk5ODY1NDE"
    isl_work = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/MUF3dWtPNjkyNzcyMDY0MTU/NjhFVkRYTzE4MjA1NTQ2OTYy/MVhpQkd2NjgzNzEwMTAzMjY"
    quetta_leg_2 = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/NldvVFZDMjk1NDg0NjA2MDg/ODdRZlVyeTM3NTIxNDAwOTE2/MlhwRnJzMjQ5MzYxNzU4ODY"
    fsl_work = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/NEN5ZWluMDIzMjU4NczM4MTA/ODROZFptdDY4NTAzMjE5Nzky/MXNkUUl3MjgwOTMwODk1NjQ"
    lahore_work = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/MlVHRGNiMjg5ODcwNTE0NDE/NzhqRHJvbDc3MTYzNDU5MDgx/MWpyWm1UNDUwOTI0MDU2NjM"
    mul_work = "https://blsitalypakistan.com/bls_appmnt/bls-italy-appointment/NUVkTklMOTY4MDA1NjE5Mzg/ODZNTGtZdjM0MDQxODkyNTE3/MUxvVGpaMTM2ODEyODc5NjQ"
    if (center == "lah_study"):
        cen = Lahore_study
    elif(center == "isl_study"):
        cen = isl_study
    elif(center == "isl_work"):
        cen = isl_work
    elif(center == "lah_work"):
        cen = lahore_work
    elif(center == "fsl_work"):
        cen = fsl_work
    elif(center == "mul_work"):
        cen = mul_work
    elif(center == "que_leg"):
        cen = Quetta
    else:
        cen = isl_study
    #driver.request_interceptor = interceptor
    current_time = time.strftime("%H:%M:%S")
    log_messages.append(f"Current time : {current_time}")
    print("Current Time:", current_time)
    while True:
        start_time = int(time.time() * 1000)
        driver.execute_cdp_cmd("Network.enable", {})

        # Define the URLs to block
        urls_to_block = ["https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit2",
                    "https://blsitalypakistan.com/common/css/font-awesome.css",
                    "*recaptcha*",
                    "https://www.google.com/recaptcha/api.js",
                    "*google-analytics*",
                    "https://blsitalypakistan.com/common/js/owl.carousel.min.js",
                    "https://blsitalypakistan.com/common/css/font-awesome.css",
                    #"*https://blsitalypakistan.com/common/css/*"
                    ]

        # Set up request interception
        driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": urls_to_block})
        driver.get(cen)
        if (driver.current_url == cen):

            try:
                #driver.execute_script("window.scrollTo(0, 800)")


                #dashboard = driver.find_element(By.XPATH, "//div[@class='dashboard']").click()
                try:
                    calendar = WebDriverWait(driver, 3).until(
                        EC.visibility_of_element_located((By.ID, "valAppointmentDate"))
                    )
                    driver.execute_script("arguments[0].scrollIntoView(true);", calendar)
                    driver.execute_script("arguments[0].click();", calendar)
                    driver.execute_script("arguments[0].focus();", calendar)
                    

                    end_time = int(time.time() * 1000)
                    loading_time = end_time - start_time
                    
                    print(f"Execution time: {loading_time} milliseconds")
                except:
                    return False
                #calendar.click()
                #driver.find_element(By.XPATH, "//div[@class='datepicker datepicker-dropdown dropdown-menu datepicker-orient-left datepicker-orient-bottom']").click()
                #input_element = label_element.find_element(By.TAG_NAME, "input")
                for i in range(col,col+2):
                    css_selector = f"tbody tr:nth-child({row}) td:nth-child({i})"
                    print(i)
                    check_start = int(time.time() * 1000)
                    bg_color_script= f"""
                    var element = document.querySelector('{css_selector}');
                    return element.getAttribute('title');
                    """

                    title = driver.execute_script(bg_color_script)
                    print("Title:", title)

                    if title == "Available":
                        log_messages.append(f"Page Loading time: {loading_time} milliseconds")
                        #find_start = int(time.time() * 1000)
    
                        script = f"""
                        var element = document.querySelector('{css_selector}');
                        return element ? element : 'Element not found';
                        """
                        date = driver.execute_script(script)
                        
                        solve_cap(driver)
                        #click_start = int(time.time() * 1000)
                        driver.execute_script("arguments[0].scrollIntoView(true);", date)
                        driver.execute_script("arguments[0].click();", date)
                        #driver.execute_script("arguments[0].focus();", date)

                        check_fail = int(time.time() * 1000)
                        c_check_time = check_fail - check_start
                        try:
                            log_messages.append(f"cap time: {c_check_time} milliseconds")
                        except:
                            print("not appended")
                        print("\n")
                        return True
                    
                print("\n")
            except Exception as e:
                print("checking for appointment error")
                print(e)
        else:
            print("logged out")
            print("\n\n")
            return False        


def payment(driver):
    time.sleep(1)
    try:
        log_messages.append(driver.current_url)
    except:
        print("appending failure")
    pb.push_note(f"Appointment open. {first_name}", text)
    pb.push_note(driver.current_url, text)
    log_messages.append(driver.current_url)
    playsound(os.path.join(base_path, "pyro.mp3"))
    print("after push")
    card_input= WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "cardNumber"))
        )
    pb.push_note(driver.current_url, text)
    print("not going forward")
    time.sleep(240)
    card_num = "" # enter your card number
    driver.execute_script("arguments[0].click();", card_input)
    driver.execute_script("arguments[0].setAttribute('Card number', '');", card_input)
    driver.execute_script("arguments[0].value = arguments[1];", card_input, card_num)
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", card_input)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", card_input)
    

    month_dropdown = driver.find_element(By.CSS_SELECTOR, 'input.select-dropdown')
    driver.execute_script("arguments[0].click();", month_dropdown)

    m_options_container = driver.find_element(By.ID, month_dropdown.get_attribute('data-activates'))
    exp_month = "" #Enter the expiry month.
    month = m_options_container.find_element(By.XPATH, f"//span[text()='{exp_month}']")
    driver.execute_script("arguments[0].click();", month)
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", month)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", month)
    time.sleep(0.1)

    year_dropdown = driver.find_element(By.XPATH, "//input[@value='2024']")
    driver.execute_script("arguments[0].click();", year_dropdown)
    y_options_container = driver.find_element(By.ID, month_dropdown.get_attribute('data-activates'))
    exp_year = "" # Enter the expiry year.
    year = y_options_container.find_element(By.XPATH, f"//span[text()='{exp_year}']")
    driver.execute_script("arguments[0].click();", year)
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", year_dropdown)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", year_dropdown)
    time.sleep(0.1)

    cvc = driver.find_element(By.XPATH, "//input[@id='ValidationCode']")
    driver.execute_script("arguments[0].click();", cvc)
    cvc_num = "" # Enter your CVC code.
    driver.execute_script("arguments[0].value = arguments[1];", cvc, cvc_num)
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", cvc)
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", cvc)
    time.sleep(0.1)

    # AFTER THE FORM HAS BEEN FILLED AUTOMATICALLY, THE USER HAS TO ONLY CLICK THE PAY BUTTON.

def book_appointment(driver):
    try:
        start_time = int(time.time() * 1000)
        app_type = WebDriverWait(driver, 6).until(
            EC.visibility_of_element_located((By.ID, "valAppointmentType"))
        )
        endin_time = int(time.time() * 1000)
        booking_time = endin_time - start_time
 
        print(f"processing time: {booking_time} milliseconds", text)
        log_messages.append(f"Processing time: {booking_time} milliseconds")

    except Exception as e:
        print(e)
        print("valappointment type not visible")
        log_messages.append(f"val appointment not visible, {first_name}")
        pb.push_note(f"val appointment not visible, {first_name}", text)
        return True

    try:
        script = f"arguments[0].selectedIndex = 1; arguments[0].dispatchEvent(new Event('change'));"
        driver.execute_script(script, app_type)
    except:
        print("len < 1")

    try:
        elements_to_fill = [
            ("input[placeholder='Enter First Name'][name='valApplicant[1][first_name]']", f"{first_name}"),
            ("input[placeholder='Enter Last Name'][name='valApplicant[1][last_name]']", f"{last_name}"),
            ("input[placeholder='Enter Nulla Osta Protocol number'][name='protocol_number']", f"{null_osta}")
        ]
                   
        script = """
        function setValueAndTriggerEvents(selector, value) {
            const element = document.querySelector(selector);
            if (element) {
                element.value = value;
                element.dispatchEvent(new Event('input', { bubbles: true }));
                element.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            }
            return false;
        }
        
        const results = {};
        arguments[0].forEach(([selector, value]) => {
            results[selector] = setValueAndTriggerEvents(selector, value);
        });
        return results;
        """
        driver.execute_script(script, elements_to_fill)
    
        check = """
        var element = document.getElementById('agree');
        return element ? element : 'Element not found';
        """

        # Execute JavaScript and get the result
        check_box = driver.execute_script(check)

        check_box_click = '''
        var element = arguments[0];
        if (element) {
            element.scrollIntoView(true);  
            element.click();              
        }
        '''
        driver.execute_script(check_box_click, check_box)
    except:
        print("name or check box")
    try:        
        b_button_script = f"""
        var element = document.querySelector('#valBookNow');
        return element ? element : 'Element not found';
        """
        button = driver.execute_script(b_button_script)
        button_click = '''
        var element = arguments[0];
        if (element) {
            element.scrollIntoView(true);  
            element.click();              
        }
        '''
        driver.execute_script(button_click, button)

        # NOW THE USER HAS TO MANUALLY CLICK THE BOOK BUTTON, AFTER WHICH THE SITE IS REDIRECTED TO THE PAYMENT PAGE.

        print(driver.current_url)
        return payment(driver)
    
    except Exception as e:
        driver.save_screenshot(f'{start_time}.png')
        print(e)
        print("returning false")
        return False
    

def login_check_book(driver):
    prev_time = time.time() - 70
    log_path = os.path.join(base_path, "processing_log.txt")
    while True:
        current_time = time.time()
        print(f"login time is :{current_time - prev_time}")
        if (current_time - prev_time) < 30:
            return True
        login(driver)
        app = check_for_appointment(driver)
        print(f"app returned {app}")
        if app == True:
            book_return = book_appointment(driver)
            if book_return == True:
                app = check_for_appointment(driver)
                if app == True:
                    book_appointment(driver)
            

        with open(log_path, 'a') as log_file:
            for message in log_messages:
                log_file.write(message + '\n')
        prev_time = current_time




def clear_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Iterate over all files and subdirectories
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # Remove the file
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # Remove the directory and its contents
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Example usage

def main():
    t = time.localtime()
    dir = os.path.join(base_path, "log1")
    options = Options()

    #options.add_argument("--no-sandbox")
    #options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-data-dir={dir}")
    options.page_load_strategy = 'eager' 
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "browser.cache.disk.enable": True,
        "browser.cache.disk.capacity": 4096
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    # Add your Chrome options here (like headless mode, etc.)
    iter = 0
    logout_error_val = 0
    while True:
        driver = webdriver.Chrome(options=options)
        driver.get("https://blsitalypakistan.com/account/login")

        cache_error = login_check_book(driver)
        if cache_error == True:
            driver.quit()
            clear_directory(dir)

if __name__ == "__main__":
    main()
