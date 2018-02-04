from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getroad',methods=['POST','GET'])
def getroad():
    if request.method=='POST':
        result=request.form

        # get inputs from user : 
        agg = result['agg']
        inter = result['int']
        catr = result['catr']
        circ=result['circ']
        vosp = result['vosp']
        prof = result['prof']
        plan = result['plan']
        lartpc = result['lartpc']
        larrout = result['larrout']
        infra = result['infra']
        situ = result['situ']
        obs = result['obs']


       

        #load necessary data : 
        df_france=pd.read_csv('dataset/df_france.csv', encoding='latin-1')
        df_bretagne=pd.read_csv('dataset/df_bretagne.csv', encoding='latin-1')

        # necessary transformations :  
        cat_vector = np.zeros(12) # user's inputs as numpy vector
        cat_vector[0]=int(agg)
        cat_vector[1]=int(inter)
        cat_vector[2]=int(catr)
        cat_vector[3]=int(circ)
        cat_vector[4]=int(vosp)
        cat_vector[5]=int(prof)
        cat_vector[6]=int(plan)
        cat_vector[7]=float(lartpc)
        cat_vector[8]=float(larrout)
        cat_vector[8]=10*(cat_vector[8]-df_france.larrout.min())/(df_france.larrout.max()-df_france.larrout.min()) # match with models features for larrout ( normalisation)
        cat_vector[9]=int(infra)
        cat_vector[10]=int(situ)
        cat_vector[11]=int(obs)

        # dict : will be used for formatting outputs to user  
        data={'agg':int(agg),'inter':int(inter),'catr':int(catr),'circ':int(circ),'vosp':int(vosp),'prof':int(prof),'plan':int(plan),'lartpc':float(lartpc),'larrout':float(larrout),'infra':int(infra),'situ':int(situ),'obs':int(obs)}

        # transforme cat_vector to a Series : 
        cat_vector_series=pd.Series(data=cat_vector,index=['agg','inter','catr','circ','vosp','prof','plan','lartpc','larrout','infra','situ','obs'])

       
        
        # needed functions :
        def amengament(instance,n_voisins=219):
            '''
            cette fonction propose un améngement pour une route très dangereuse.
            Elle permet de chercher dans la base de la France les plus proches voisins dont le risque de gravité est quasiment nulle
            '''
            ###### trouvons les plus proches voisin à cette route en utilisant KNN de la gravité : 
            voisins= knn.kneighbors(instance.reshape(1, -1),n_neighbors=n_voisins,return_distance=False)
            res=pd.DataFrame()
            for j in range(n_voisins) :
                i=voisins[0][j] # get index of the first neighbor for the instance used for validation ( here j=0)
                risk_road=df_france.loc[i,'risk_gravity']  # get classification of the neighbor
                if risk_road==1:
                    my_dict=df_france.loc[i,col_train].to_dict()
                    del my_dict['risk_gravity']
                    del my_dict['risk_frequence']
                    df=pd.DataFrame(my_dict,index=[0])
                    res=pd.concat([res,df])            
            # create index : 
            res=res.reset_index()
            res=res.iloc[:,1:]
            return res


        def prediction_gravity_frequence(new_instance,n_voisins=219):
            '''
            Parameters : 
                - new_instance : la route saisie par un utilisateur 
                - n_voisins : nombre de voisins utilisés pour proposer un aménagement
    
    
            Returns : 
                -  deux trames de données : une pour les résultats sur les scores et une pour l'aménagement
    
    
    
            '''
            #gravity : 
            prediction_gravity= knn.predict(new_instance.reshape(1, -1))
            prediction_gravity_proba= knn.predict_proba(new_instance.reshape(1, -1) )
    
            #frequence : 
            prediction_frequence= knn_frequence.predict(new_instance.reshape(1, -1) )
            prediction_frequence_proba= knn_frequence.predict_proba(new_instance.reshape(1, -1) )
            #results into a dataframe  :
            my_dict=new_instance.to_dict()
            res=pd.DataFrame(my_dict,index=[0])
            #gravity results : 
            if prediction_gravity[0]==1 :
                res['la gravité']='Route non dangereuse'
            else : 
                res['la gravité']='Route très dangereuse'
            res['probabilité de la gravité']=prediction_gravity_proba[0][prediction_gravity-1][0]
            #frequence results :  
            if prediction_frequence[0]==1 :
                res['la fréquence des accidents']='accidents non fréquents'
            else : 
                res['la fréquence des accidents']='accidents très fréquents'
            res['probabilité de la fréquence']=prediction_frequence_proba[0][prediction_frequence-1][0]
            solution=pd.DataFrame()
            solution=amengament(new_instance,n_voisins)
        
        
            return res,solution




        #load saved models : 
        pkl_file = open('pkl_objects/knn.pkl', 'rb')
        knn = pickle.load(pkl_file)
        pkl_file2 = open('pkl_objects/knn_frequence.pkl', 'rb')
        knn_frequence = pickle.load(pkl_file2)
        # training columns + targets
        col_train =[ 'agg', 'int', 'catr', 'circ', 'vosp', 'prof', 'plan', 'lartpc','larrout' ,'infra', 'situ', 'obs', 'risk_gravity','risk_frequence']
        validation=df_bretagne.loc[:,col_train]
        
        # instance : route saisie par l'utilisateurs 
        instance=cat_vector_series   # retirer les variables cibles 

        #resultas = prediction_gravity_frequence(instance,n_voisins=219)[0]
        #resultas=resultas.to_html()

        amengament_propose = prediction_gravity_frequence(instance,n_voisins=219)[1]
        #amengament_propose=amengament_propose.to_html()
        prediction_gravity = int(knn.predict(cat_vector))
        proba_gravity=round(knn.predict_proba(cat_vector.reshape(1, -1))[0][prediction_gravity-1],2)

        prediction_frequence = int(knn_frequence.predict(cat_vector))
        proba_frequence=round(knn_frequence.predict_proba(cat_vector.reshape(1, -1))[0][prediction_frequence-1],2)

        # dict used for formatting outputs to user interface
        dict_gravity={1:'Route sûre',2:'Route très dangereuse'}
        dict_frequence={1:'Accidents non fréquents',2:'Accidents très fréquents'}
 
        dict_agg={0:'sans objet', 1:'Hors agglomération',2:'En agglomération'}
        dict_inter={0:'sans objet',1:'hors intersection',2:'intersection en X',3:'intersection en T',4:'intersection en Y',5:'intersection à plus de 4 branches' ,6:'giratoire',7:'place',8:'Passage à  niveau',9:'autre intersection'}
        dict_catr={0:'sans objet',1:'Autoroute',2:'Route Nationale',3:'Route Départementale',4:'Voie Communale',5:'Hors réseau public',6:'Parc de stationnement ouvert à la circulation public',9:'Autre'}
        dict_circ={0:'sans objet',1:'à sens unique',2:'bidirectionnelle',3:'à chaussées séparées',4:'avec voie d\'affecation variables'}
        dict_vosp={0:'sans objet',1:'piste cyclable',2:'banque cyclable',3:'voie résérvée'}
        dict_prof={0:'sans objet',1:'plat',2:'pente',3:'sommet de côte',4:'bas de coûte'}
        dict_plan={0:'sans objet',1:'partie rectiligne',2:'en courbe à gauche',3:'en courbe à droite',4:'en "S" '}
        dict_infra={0:'sans objet',1:'souterrain-tunnel',2:'pont-autopont',3:'bretelle d\'échangeur',4:'voie férrée',5:'carrefour aménagé',6:'zone piéton',7:'zone de péage'}
        dict_situ={0:'sans objet',1:'sur chaussée',2:'sur bande d\'arrêt d\'urgence',3:'sur accotement',4:'sur trottoir',5:'sur piste cyclable'}
        dict_obs={0:'sans objet',1:'Véhicule en stationnement',2:'Arbre',3:'Glissière métallique',4:'Glissière béton',5:'Autre glissièr',6:'Bâtiment, mur, pile de pont',
        7:'Support de signalisation verticale ou poste d’appel d’urgence ',8:'Poteau',9:'Mobilier urbain ',10:'Parapet',11:'Ilot, refuge, borne haute',12:'Bordure de trottoir ',
        13:'Fossé, talus, paroi rocheuse',14:'Autre obstacle fixe sur chaussée',15:'Autre obstacle fixe sur trottoir ou accotement',16:'Sortie de chaussée sans obstacle '}

        amengament_propose.columns=['Localisation','Catégorie de route','Régime de circulation','Aménagement infrastructure','Type d\'intersection',
        'Largeur de la chaussée(mètres)','Largeur de terre plein centrale (mètres)','Obstacle fixe heurté','Tracé en plan',
        'Profil de la route', 'Situation de l\'accident','Existence d\'une voie résérvée']
        
        amengament_propose['Localisation']=amengament_propose['Localisation'].map(dict_agg)
        amengament_propose['Type d\'intersection']=amengament_propose['Type d\'intersection'].map(dict_inter)
        amengament_propose['Catégorie de route']=amengament_propose['Catégorie de route'].map(dict_catr)
        amengament_propose['Régime de circulation']=amengament_propose['Régime de circulation'].map(dict_circ)
        amengament_propose['Existence d\'une voie résérvée']=amengament_propose['Existence d\'une voie résérvée'].map(dict_vosp)
        amengament_propose['Profil de la route']=amengament_propose['Profil de la route'].map(dict_prof)
        amengament_propose['Tracé en plan']=amengament_propose['Tracé en plan'].map(dict_plan)
        amengament_propose['Aménagement infrastructure']=amengament_propose['Aménagement infrastructure'].map(dict_infra)
        amengament_propose['Situation de l\'accident']=amengament_propose['Situation de l\'accident'].map(dict_situ)
        amengament_propose['Obstacle fixe heurté']=amengament_propose['Obstacle fixe heurté'].map(dict_obs)
        amengament_propose['Largeur de la chaussée(mètres)']=(1/10)*amengament_propose['Largeur de la chaussée(mètres)']*((df_france.larrout.max()-df_france.larrout.min()))+df_france.larrout.min()


        return render_template('result.html',tables=[amengament_propose.to_html()],data=data,dict_gravity=dict_gravity,dict_frequence=dict_frequence,
            prediction_gravity=prediction_gravity,proba_gravity=proba_gravity,prediction_frequence=prediction_frequence,proba_frequence=proba_frequence,
            dict_agg=dict_agg,dict_inter=dict_inter,dict_catr=dict_catr,dict_circ=dict_circ,dict_vosp=dict_vosp,dict_prof=dict_prof,dict_plan=dict_plan,dict_infra=dict_inter,dict_situ=dict_situ,dict_obs=dict_obs)

    
if __name__ == '__main__':
	app.debug = True
	app.run()