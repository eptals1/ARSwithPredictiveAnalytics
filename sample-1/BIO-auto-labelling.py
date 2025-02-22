import pandas as pd
import nltk

# Ensure you have the tokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Define the label mappings
label_map = {
    "Age": "AGE",
    "Gender": "GEN",
    "Address": "LOC",
    "Skills": "SKILL",
    "Education": "EDU",
    "Experience": "EXP",
    "Certificates": "CERT"
}

# Load the dataset
data = """
Age,Gender,Address,Skills,Education,Experience,Certificates
Age,Gender,Address,Skills,Education,Experience,Certificates
25,Male,"P1, Ong Yiu, Butuan City",Computer Programming,College Graduate,Backend Web Developer,Web Dev NCIII
23,Male,"Ong Yiu, Butuan City",Driving,Senior High School Graduate,Driving,Driving NCI
31,Female,P3A Baan Km3 Ora New Road,communication,College Graduate,"CASHIER VROSS, HAPPY ENTERPRISES, GMALL, WING ON",NONE
22,Female,Doongan,Communication ,College Undergraduate,Virtual Assistant ,LTS
21,Female,P-7 Dumalagan Butuan City ,Graphics Design,College Undergraduate,"Encoder, Visual Graphic ","2nd Place in Visual Graphic Technology at Northern Mindanao, Buenavista, Agusan Del Northe"
20,Male,"Purok 8, Brgy.Bitan-agan, Butuan City, Agusan del Norte ",Video Editing ,College Undergraduate,Driver,NCI
26,Female,Bitan Agan Butuan Cirty,Computer Literate,College Graduate,Site Engineer ,N/A
22,Female,"Purok Nangka, Barangay 6, Buenavista, Agusan del Norte ",Computer Literate ,College Graduate,Private School Teacher ,Certificate of Eligibility 
26,Female,Brgy J.P Rizal,Computer Literate,College Undergraduate,Costumer Service Officer,none
19,Female,P-5 Bitan agan Butuan City,Video editing,College Undergraduate,none,None
25,Male,Purok Nangka Barangay 6 Buenavista Agusan del Norte,Computer Literate,College Graduate,Czarles Construction HR,Web developer 
21,Female,P-3 Amparo Butuan City,Video editing ,Senior High School Graduate,Cashier,None
22,Male,"P-8 Maug, Butuan City",Video Editing ,College Undergraduate,None,None
19,Female,Vutuan city,Communication,College Undergraduate,None,None
21,Female,P-6 Bitan-agan Butuan City ,"Graphics Design, Communication ",College Undergraduate,Web development ,Web development 
19,Male,R.T.R Balangbalang purok-1,Video Editing,College Undergraduate,None,None
22,Male,Bitan-Agan Butuan City,Video Editing ,College Undergraduate,None,None
23,Female,P-6 Bitan-agan Butuan City ,Communication ,College Undergraduate,Casher,Cookery 
19,Female,"P-5 Bitan-agan, Butuan City",communications,College Undergraduate,none,none
20,Male,Purok 4 limaha,Video editing,College Undergraduate,"Janitor, assistant construction worker, kasambahay ",None
16,Female,"Anunas, Angeles City Pampanga ",Communications,Junior High School Graduate,None,None
22,Female,"Purok 4 Amparo, Butuan City",Video Editing and Graphics Design ,College Undergraduate,Web Development ,None
21,Male,Ecoland Davao city ,"Video editor, communication ",College Undergraduate,"Welder, electrician, plumbing,installer",N/A
20,Female,"Bitan-agan, Butuan City ",Video editing and communication ,College Undergraduate,None,None
22,Female,P9 upper Doongan ,Video editing ,Junior High School Graduate,"Cashier, Graphic Artist",None
21,Female,"P-5A Amparo, Butuan City ",Graphics Design,College Undergraduate,N/A,N/A
25,Male,"P-5 Ambago, Butuan City",Graphics Design,College Undergraduate,Painters,Cyber Security Sentinel
25,Female,"P-1, Brgy. 19 Buhangin, Butuan City","Graphics Design, Video Editing, Computer Literate",College Graduate,Bookkeeper/Accounting Clerk,Certified Xero Advisor
26,Female,Km.3 Baan Butuan City ,Communications,Senior High School Graduate,Cashier ,None
40,Female,P-4 Ampayon Butuan City,Computer literate,College Graduate,Office staff,None
24,Female,Amparo butuan City ,Graphics ,Technical Vocational Graduate,Merchandiser ,None
32,Male,Butuan City,"Graphic Design, Photo and Video editing",College Graduate,Graphic Artist,N/A
23,Female,Los Angeles Butuan City ,Computer Literate ,College Graduate,"Cashier, Encoder",Agricultural Crops Production NCI
22,Female,T Calo Muslim appears ,I.T,College Undergraduate,"Cashier,Oic, B.A",Cookery
31,Female,Buenavista Agusan Del Norte,Computer Literate,College Graduate,Bank compliance officer II,AMLA training Certificate
31,Female,Barangay 8 Buenavista Agusan Del Norte ,Video,Elementary Graduate,Housekeeper ,NCI
31,Female,Barangay 8 Buenavista Agusan Del Norte ,Video,Elementary Graduate,Housekeeper ,NCI
38,Female,"P-3B Manapa Buenavista, Agusan del Norte","Communication, computer literate",College Graduate,"Teacher, HR officer, Project coordinator",N/A
35,Female,Upper macalang buenavista agusan del norte,Communications ,Junior High School Graduate,Etc...,Etc..
38,Female,Brgy. 1 District Adelfa Apagan Nasipit Agusan del Norte,video editing ,College Graduate,Secretary ,N/A
18,Female,"P-3B Manapa, Buenavista Agusan Del Norte ",Volleyball ,Senior High School Graduate,Cashier ,None
30,Female,Purok 6 macalang Buenavista Agusan del norte ,None of the above ,Junior High School Graduate,None of the above ,None
30,Female,Simbalan Buenavista Agusan del Norte,Housewife,Elementary Graduate,Housewife,No
31,Female,"Purok-6 Macalang, Buenavista agusan del Norte ",Cooking ,Junior High School Graduate,Field Sales Agent,N/A
36,Female,Purok-3A Manapa Buenavista Agusan del Norte ,None ,College Undergraduate,Cashier ,NC2
34,Female,Buenavista agusan del norte,computer,Senior High School Graduate,Casher,Nc11

"""

# Convert to DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# Function to convert row into NER format
def convert_row_to_ner(row):
    output = []
    
    for col in df.columns:
        if pd.notna(row[col]):  # Ignore empty values
            tokens = word_tokenize(str(row[col]))  # Tokenize words
            entity_type = label_map[col]  # Get entity type

            for i, token in enumerate(tokens):
                label = f"B-{entity_type}" if i == 0 else f"I-{entity_type}"
                output.append(f"{token} {label}")
    
    return output

# Convert all rows
ner_formatted_data = []
for _, row in df.iterrows():
    ner_formatted_data.extend(convert_row_to_ner(row))
    ner_formatted_data.append("")  # Add a newline between resumes

# Save to file
with open("ner_data.txt", "w") as f:
    f.write("\n".join(ner_formatted_data))

# Print sample output
print("\n".join(ner_formatted_data[:20]))  # Show only first 20 lines
