Chapter - 1: Introduction

In the fast-paced realm of technological advancements, organizations are increasingly seeking efficient and secure alternatives to traditional attendance tracking systems. The Face Recognition Attendance System stands out as a revolutionary solution, leveraging cutting-edge technology to redefine how attendance is managed. Conventional methods, relying on manual processes like sign-in sheets or card-swiping, are not only time-intensive but also susceptible to errors such as buddy punching and lost identification cards. The Face Recognition Attendance System addresses these challenges by harnessing facial recognition algorithms and artificial intelligence.

Facial recognition technology has progressed significantly, allowing systems to accurately identify individuals based on distinct facial features. Deep learning models enable precise identification in dynamic environments with varying lighting conditions. The emphasis on efficiency, accuracy, and security in attendance management positions this system as a leading-edge solution. Its benefits extend beyond technical capabilities, offering real-time tracking, reduced administrative workload, and enhanced user convenience.

This introduction sets the stage for a detailed exploration of the Face Recognition Attendance System, covering its features, technological foundations, and transformative impact on organizational processes. As we delve into this innovative technology, it becomes clear that it not only modernizes attendance tracking but also signifies a shift towards smarter, more secure workplace management systems, aligning with the evolving needs of organizations. 
Chapter - 2: Project Aim

The primary aim of the Face Recognition Attendance System is to modernize and optimize the traditional methods of attendance tracking within organizations. This advanced system harnesses facial recognition technology to achieve several key objectives:

Accuracy and Precision:
Ensure accurate and precise identification of individuals based on their unique facial features, reducing the likelihood of errors associated with manual attendance processes.

Efficiency and Time Savings:
Streamline the attendance tracking process by automating the identification and recording of individuals in real-time, eliminating the need for manual data entry and reducing administrative workload.

Security Enhancement:
Enhance the security of attendance data by utilizing advanced facial recognition algorithms and encryption techniques, safeguarding against unauthorized access and ensuring the integrity of attendance records.

Contactless and Convenient:
Provide a contactless and convenient alternative to traditional attendance methods, contributing to a more hygienic and user-friendly experience for individuals in various organizational settings.

Real-time Monitoring:
Enable real-time monitoring of attendance, allowing organizations to have instantaneous access to attendance data for quick decision-making and efficient resource allocation.

Integration with Existing Systems:
Facilitate seamless integration with existing attendance management systems, ensuring compatibility with diverse organizational structures and technology infrastructures.


Reduction of Fraudulent Practices:
Mitigate the risk of fraudulent practices such as buddy punching by relying on biometric identification, ensuring that attendance records accurately reflect individual presence.

User-Friendly Experience:
Offer a user-friendly interface for both administrators and end-users, promoting ease of use and encouraging widespread adoption within the organization.

Compliance with Regulations:
Adhere to privacy and ethical guidelines, complying with data protection regulations and prioritizing the responsible and ethical use of facial recognition technology.

In summary, the Face Recognition Attendance System aims to revolutionize attendance tracking by leveraging advanced technology to enhance accuracy, efficiency, security, and user experience within organizational contexts. 
Chapter - 3: Problem Formulation

Problem statement:


Traditional methods of attendance tracking in organizations, such as manual sign-in sheets or card-based systems, are plagued with several inherent challenges that hinder efficiency, accuracy, and security. The need for a more sophisticated and reliable solution has given rise to the development of Face Recognition Attendance Systems (FRAS). However, the adoption of FRAS is not without its own set of challenges and concerns. This problem statement identifies key issues associated with the implementation and usage of face recognition technology for attendance management: 
PROBLEMS:

•	Accuracy and Reliability:
Facial recognition systems may encounter difficulties in accurately identifying individuals under varying lighting conditions, different facial expressions, or with the presence of accessories (e.g., glasses, hats). Ensuring consistent and reliable performance remains a critical challenge.

•	Data Security and Privacy Concerns:
The use of biometric data, specifically facial features, raises significant concerns regarding data security and privacy. Protecting this sensitive information from unauthorized access and ensuring compliance with privacy regulations pose complex challenges.

•	Integration with Existing Systems:
Integrating FRAS with pre-existing attendance management systems or other organizational databases can be challenging. Compatibility issues and the seamless transfer of data need to be addressed for successful integration.

•	Cost Implications:
The initial setup and deployment costs of implementing a Face Recognition Attendance System can be a barrier to adoption for smaller organizations with limited budgets. The cost-effectiveness and return on investment need to be carefully considered.

•	User Acceptance and Training:
Users may be resistant to adopting new technology, and a lack of understanding or training regarding the use of FRAS could impede its successful implementation. Ensuring user acceptance and providing comprehensive training become crucial factors.






•	Ethical Considerations:
Ethical concerns surrounding the use of facial recognition technology, including issues related to consent, surveillance, and potential misuse, need to be thoroughly addressed to garner support and trust from both employees and stakeholders.

•	Environmental Factors:
Environmental factors such as varying lighting conditions, camera quality, and the physical placement of devices can impact the performance of FRAS. Developing systems that are robust in different environments is a significant challenge.

•	Legal and Regulatory Compliance:
Keeping abreast of evolving legal and regulatory frameworks related to facial recognition technology is imperative. Ensuring compliance with data protection laws and industry standards is an ongoing challenge.
This problem statement identifies critical issues that organizations and developers must navigate when implementing a Face Recognition Attendance System, emphasizing the need for comprehensive solutions that address technical, ethical, and regulatory considerations.

 
Chapter - 4. Literature Survey-

S NO	RESEARCH PAPER	CONCLUSION OF RESEARCH PAPER	OUR ADDITION





 01	


International Journal 
of 
Engineering        Research   
& 
Technology (IJERT)
Issued on  
05, May-2020	The proposed face recognition-based class attendance system aims to address the time-consuming and error-prone manual attendance methods. It leverages the advantages of face recognition, offering a contactless and non-invasive process. The system demonstrates efficiency in marking attendance by detecting and recognizing faces in live streaming videos.	Multi-Algorithm Fusion: 
The integration of multiple face recognition algorithms (e.g., distance and similarity algorithm like Euclidian , Manhattan, Cosine) and utilize a fusion approach. Combining the strengths of different algorithms can enhance overall system robustness.







02
	



International Journal 
for 
Innovative  Engineering 
And Management Research
Issued on  
03, Mar-2023	The paper presents screenshots of the system's user interface, showcasing data collection, image capture, and attendance marking. Algorithms like Haar Cascade and LBPH are employed for face detection and recognition. The proposed system effectively recognizes faces and records attendance in real-time, offering a viable alternative to manual methods.	Advanced Face Recognition Algorithms:
Explore and implement state-of-the-art face recognition algorithms. Keep abreast of advancements in the field, such as deep learning-based models like FaceNet,   Insightface, which may offer improved accuracy
 
Chapter -5: Methodology

Libraries used in this project.
1.	Numpy
2.	Pandas
3.	OpenCV
4.	Insightface
5.	Sklearn
6.	Matplotlib
7.	Streamlit
8.	Av & time
9.	Jupyter
10. Scipy

We implemented multiple page web app based on our predefined file
Face_rec.py
import numpy as np
import pandas as pd
import cv2

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
# time
import time
from datetime import datetime

import os

# Connect to Redis Client
hostname = 'redis-17525.c264.ap-south-1-1.ec2.cloud.redislabs.com'
portnumber = 17525
password = 'Deeksha@17'


r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

# Retrive Data from database
def retrive_data(name):
    retrive_dict= r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df =  retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role','facial_features']
    retrive_df[['Name','Role']] = retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name','Role','facial_features']]


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_l',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,
                        name_role=['Name','Role'],thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
        
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    return person_name, person_role


### Real Time Prediction
# we need to save logs for every 1 mins
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def saveLogs_redis(self):
        # step-1: create a logs dataframe
        dataframe = pd.DataFrame(self.logs)        
        # step-2: drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name',inplace=True) 
        # step-3: push data to redis database (list)
        # encode the data
        name_list = dataframe['name'].tolist()
       
role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
                
        if len(encoded_data) >0:
            r.lpush('attendance:logs',*encoded_data)
        
                    
        self.reset_dict()     
        
        
    def face_prediction(self,test_image, dataframe,feature_column,
                            name_role=['Name','Role'],thresh=0.5):
        # step-1: find the time
        current_time = str(datetime.now())
        
        # step-1: take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        # step-2: use for loop and extract each embedding and pass to ml_search_algorithm

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            if person_name == 'Unknown':
                color =(0,0,255) # bgr
            

else:
                color = (0,255,0)

            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
            

        return test_copy


#### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
        
    def get_embedding(self,frame):
        # get results from insightface model
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            

# put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            
            # facial features
            embeddings = res['embedding']
            
        return frame, embeddings
    
    def save_data_in_redis_db(self,name,role):
        # validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        # if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        
        
        # step-1: load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32) # flatten array            
        
        # step-2: convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)       
        
        # step-3: cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        
        # step-4: save this into redis database
        # redis hashes
        r.hset(name='college:register',key=key,value=x_mean_bytes)
        
        # 
        os.remove('face_embedding.txt')
        self.reset()
        
        return True

 
SAMPLES-

        GRAPH: 
        ![image](https://github.com/user-attachments/assets/8318959f-15f9-49d9-ba67-8e63d63b3142)

Outlook of web app
•	HOME
![image](https://github.com/user-attachments/assets/056213d6-edea-49cb-8f34-99dda7464ba1)

•	REALTIME ATTENDANCE 
![image](https://github.com/user-attachments/assets/3600edc2-c5e4-400e-ba80-35da3d06dd59)

•	REGISTRATION FORM
![image](https://github.com/user-attachments/assets/ae8e66d8-60c7-45b8-9264-7a66fef5be16)

•	REPORTING
![image](https://github.com/user-attachments/assets/97b581fa-9e47-4408-99cc-b34e76085c13)

•	PREDICTION
![image](https://github.com/user-attachments/assets/03c4a973-0d20-4e76-8e18-d076cc05463b)

5.	Conclusion
The proposed system is designed to provide an automated attendance system for lectures, sections, and laboratories, allowing lecturers or teaching assistants to easily record student attendance. By utilizing face detection and recognition algorithms, this system saves time and effort, especially in classes with a large number of students. This automated system can improve an institution's goodwill by reducing drawbacks in the traditional manual system. Through thorough testing of the face detection and recognition algorithms, student attendance is marked by recognizing their face and storing the data in an attendance sheet. The system was developed from requirements to a complete system, including evaluation and testing, and achieved its objectives to the satisfaction of the client. Although some challenges were encountered during implementation, they were addressed and resolved. Strategies for future work and improvements to the system are discussed in this section

6.	References

Kaggle.com/face_recognition_attendance_system/

International Journal of Engineering  Research & Technology (IJERT)
		Issued on 05, May-2020

International Journal for Innovative Engineering and Management     Research 
Issued on 03, Mar-2023


