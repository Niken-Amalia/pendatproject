import streamlit as st
import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import pickle

st.title("Aplikasi Web Datamining")
st.write("""
# Maternal Health Risk Dataset
Aplikasi Berbasis Web untuk Mengklasifikasi **Maternal Health Risk**,
Jadi pada web applikasi ini akan bisa membantu user untuk mengklasifikasikan sebuah dataset Maternal Health Risk atau Resiko Kesehatan Ibu,
dimana nanti user akan bisa menginputkan suatu data dari setiap fitur yang ada dalam dataset maternal health risk ini,
sehingga nanti akan dapat menemukan level resiko kesehatan pada ibu hamil dan juga anda dapat melihat akurasi
dari beberapa algoritma yang di sediakan dalam aplikasi ini, sehingga user dapat melihat serta membandingkan 
tingkat akurasi yang paling terbaik dari model algoritma yang disediakan aplikasi tersebut.
""")

# inisialisasi data 
data = pd.read_csv("Healthrisk.csv")
tab1, tab2, tab3, tab4, tab5= st.tabs(["Description Data", "Preprocessing Data", "Modeling", "Implementation", "Profil"])

with tab1:

    st.subheader("Deskripsi Dataset")
    st.write("""Dataset ini akan menjelaskan mengenai maternal health risk.Dimana merupakan dataset untuk mengetahui
    tingkal kesehatan ibu hamil.Pada Dataset ini memiliki 6 fitur yaitu age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate.
    Data telah dikumpulkan dari berbagai rumah sakit, klinik komunitas, layanan kesehatan 
    ibu melalui sistem pemantauan risiko berbasis IoT.
    """)

    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
    st.write("""
    ### Deskripsi Fitur
    Disini di jelaskan data-data yang ada dalam dataset tersebut seperti penjelasan dari setiap fitur yang
    ada dalam dataset tersebut :
    1. Age          : Usia dalam tahun ketika seorang wanita hamil..
    2. SystolicBP   : Nilai atas Tekanan Darah dalam mmHg.
    3. DiastolicBP  : Nilai Tekanan Darah yang lebih rendah dalam mmHg.
    4. BS           : Kadar glukosa darah dinyatakan dalam konsentrasi molar, mmol/L.
    5. BodyTemp     : Suhu tubuh dalam satuan Fahrenheit
    6. HeartRate    : Detak jantung istirahat normal dalam denyut per menit.
    7. RiskLevel    : Prediksi Tingkat Intensitas Risiko selama kehamilan dengan mempertimbangkan atribut sebelumnya.
    """)
    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
    st.write("""
    ### Sumber
    - Dataset [kaggel.com](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data)
    - Github Account [github.com](https://github.com/Niken-Amalia/pendatproject)
    """)

with tab2:
    st.subheader("Data Preprocessing")
    st.subheader("Data Asli")
    data = pd.read_csv("Healthrisk.csv")
    st.write(data)

    proc = st.checkbox("Normalisasi")
    if proc:

        # Min_Max Normalisasi
        from sklearn.preprocessing import MinMaxScaler
        df_for_minmax_scaler=pd.DataFrame(data, columns = ['Age',	'SystolicBP',	'DiastolicBP',	'BS',	'BodyTemp',	'HeartRate'])
        df_for_minmax_scaler.to_numpy()
        scaler = MinMaxScaler()
        df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

        st.subheader("Hasil Normalisasi Min_Max")
        df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ['Age',	'SystolicBP',	'DiastolicBP',	'BS',	'BodyTemp',	'HeartRate'])
        st.write(df_hasil_minmax_scaler)

        st.subheader("tampil data risklevel")
        df_RiskLevel = pd.DataFrame(data, columns = ['RiskLevel'])
        st.write(df_RiskLevel.head())

        st.subheader("Gabung Data")
        df_new = pd.concat([df_hasil_minmax_scaler,df_RiskLevel], axis=1)
        st.write(df_new)

        st.subheader("Drop fitur RiskLevel")
        df_drop_site = df_new.drop(['RiskLevel'], axis=1)
        st.write(df_drop_site)

        st.subheader("Hasil Preprocessing")
        df_new = pd.concat([df_hasil_minmax_scaler,df_RiskLevel], axis=1)
        st.write(df_new)

with tab3:

    X=data.iloc[:,0:6].values
    y=data.iloc[:,6].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    st.subheader("Pilih Model")
    model1 = st.checkbox("KNN")
    model2 = st.checkbox("Naive Bayes")
    model3 = st.checkbox("Random Forest")
    # model4 = st.checkbox("Ensamble Stacking")

    if model1:
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "KNN.pkl"
        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma KNN : ",score)
    if model2:
        model = GaussianNB()
        filename = "GaussianNB.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    if model3:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Random Forest : ",score)
    # if model4:
        # estimators = [
        #     ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
        #     ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        #     ]
        # model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        # filename = "stacking.pkl"

        # model.fit(X_train,y_train)
        # Y_pred = model.predict(X_test)

        # score=metrics.accuracy_score(y_test,Y_pred)
        # loaded_model = pickle.load(open(filename, 'rb'))
        # st.write("Hasil Akurasi Algoritma Ensamble Stacking : ",score)

with tab4:
    # Min_Max Normalisasi
    from sklearn.preprocessing import MinMaxScaler
    df_for_minmax_scaler=pd.DataFrame(data, columns = ['Age',	'SystolicBP',	'DiastolicBP',	'BS',	'BodyTemp',	'HeartRate'])
    df_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

    df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ['Age',	'SystolicBP',	'DiastolicBP',	'BS',	'BodyTemp',	'HeartRate'])

    df_RiskLevel = pd.DataFrame(data, columns = ['RiskLevel'])

    df_new = pd.concat([df_hasil_minmax_scaler,df_RiskLevel], axis=1)

    df_drop_site = df_new.drop(['RiskLevel'], axis=1)

    df_new = pd.concat([df_hasil_minmax_scaler,df_RiskLevel], axis=1)

    st.subheader("Parameter Inputan")
    # SEQUENCE_NAME = st.selectbox("Masukkan SEQUENCE_NAME : ", ("AAT_ECOLI","ACEA_ECOLI","ACEK_ECOLI","ACKA_ECOLI",
    # "ADI_ECOLI","ALKH_ECOLI","AMPD_ECOLI","AMY2_ECOLI","APT_ECOLI","ARAC_ECOLI"))
    Age = st.number_input("Masukkan Umur :")
    SystolicBP = st.number_input("Masukkan Tekanan Darah Systolic :")
    DiastolicBP = st.number_input("Masukkan Tekanan Darah Diastolic :")
    BS = st.number_input("Masukkan Kadar Gula Darah :")
    BodyTemp = st.number_input("Masukkan Suhu Tubuh:")
    HeartRate = st.number_input("Masukkan Detak Jantung :")
    hasil = st.button("cek klasifikasi")

    # Memakai yang sudah di preprocessing
    X=df_new.iloc[:,0:6].values
    y=df_new.iloc[:,6].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    if hasil:
        # if algoritma == "KNN":
        #     model = KNeighborsClassifier(n_neighbors=3)
        #     filename = "KNN.pkl"
        # elif algoritma == "Naive Bayes":
        #     model = GaussianNB()
        #     filename = "gaussianNB.pkl"
        # elif algoritma == "Random Forest":
        #     model = RandomForestClassifier(n_estimators = 100)
        #     filename = "RandomForest.pkl"
        # else:
        #     estimators = [
        #         ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
        #         ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        #         ]
        #     model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        #     filename = "stacking.pkl"
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        
        dataArray = [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        #st.write(f"Algoritma yang digunakan adalah = Random Forest Algorithm")
        st.success(f"Hasil Akurasi : {score}")
with tab5:

    st.subheader("Profil Mahasiswa")
    st.write("""
    \nNama     : Niken Amalia
    \nNim      : 200411100109
    \nKelas    : Data Mining A
    \nDisini saya mengerjakan project akhir dari mata kuliah Data Mining.
    \n**Data Mining** adalah suatu proses pengerukan atau pengumpulan informasi penting dari suatu data yang besar atau big data.
    \nKali ini saya mencoba melakukan analisis pada dataset yaitu Maternal Health Risk atau Resiko kesehatan ibu saat kehamilan.
    \nUntuk Pertanyaan dan info lebih lanjut kalian bisa hubungi kontak dibawah yaa !!
    \n**Email**     : nknaml05@gmail.com
    \n**Github**    : niken-amalia
    \n**Instagram** : _nikenam
    \nSemoga Bermanfaat,SEE YOU 
    """)
