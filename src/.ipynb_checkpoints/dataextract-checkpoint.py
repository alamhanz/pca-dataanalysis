import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import sys
sys.setrecursionlimit(20000)

def circleOfCorrelations(pc_infos1, ebouli, p1="PC-0",p2="PC-1",rad=1,top_quality=None,annot=False):
    pc_infos0=pc_infos1.copy()
    plt.figure(figsize=(10,10))
    circle1=plt.Circle((0,0),radius=rad, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    pc_infos0['quality']=np.sqrt(pc_infos0[p1]**2+pc_infos0[p2]**2)
    if top_quality==None:
        pc_infos=pc_infos0.copy()
    else:
        pc_infos=pc_infos0.sort_values('quality',ascending=False).head(top_quality)
    
    for idx in range(len(pc_infos[p1])):
        x = pc_infos[p1][idx]
        y = pc_infos[p2][idx]
        plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        if annot:
            plt.annotate(pc_infos.index[idx], xy=(x,y))
    plt.xlabel(p1+" (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel(p2+" (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.xlim((-rad,rad))
    plt.ylim((-rad,rad))
    plt.title("Circle of Correlations")

def clean_high_corr(all_feat,corr_mat,feat_val,bench=0.85):
    not_use=[]
    feat_high=feat_val[all_feat].sort_values(ascending=False).index.tolist()  
    for ff in feat_high:
        if ff not in not_use:
            cm=corr_mat[ff]
            cm_list=cm[cm>bench].index.tolist()
            cm_list.remove(ff)
            not_use+=cm_list
    new_m1=list(set(all_feat)-set(not_use))
    return new_m1

class data_pca_exploration:
    def __init__(self,all_clean_data,feature_focus):
        self.raw_data=all_clean_data.copy()
        self.feature_focus=feature_focus.copy()
        self.raw_data=self.raw_data[self.feature_focus]
        print(self.raw_data.shape)

    def outlier_label(self,anomalies_algorithms,normalize=True):
        ## Normalize
        if normalize:
            ## normalize it
            Xdata0=self.raw_data[self.feature_focus]
            Xdata=(Xdata0 - Xdata0.mean()) / Xdata0.std()
        else:
            Xdata=self.raw_data[self.feature_focus]
            
        ## Fitting anomaly algo
        self.anomalies_algorithms=anomalies_algorithms
        print('fitting anomaly')
        self.anomalies_algorithms=self.anomalies_algorithms.fit(Xdata)
        self.raw_data['outliers'] = self.anomalies_algorithms.predict(Xdata)
        print('done')
    
    
    def pca_features(self,principle=None,normalize=True,outliers=True):
        if principle:
            self.principle=principle
        else:
            self.principle=None
        
        if 'outliers' not in self.raw_data.columns:
            print("outlier column name doesn't exist")
            self.outlier_label(IsolationForest)
            print("Fitting outlier with Isolation Forrest")

        ## Normalize
        if normalize:
            ## normalize it
            Xdata0=self.raw_data[self.feature_focus]
            Xdata1=self.raw_data[self.raw_data.outliers!=-1][self.feature_focus]
            Xdata=(Xdata0 - Xdata1.mean()) / Xdata1.std()
        else:
            Xdata=self.raw_data[self.feature_focus]
        
        Xdata['outliers']=self.raw_data.outliers
        self.normalize_data=Xdata
        self.pca_mod = PCA(n_components=self.principle)
        if outliers:
            self.pca_mod = self.pca_mod.fit(Xdata[self.feature_focus])
        else:
            self.pca_mod = self.pca_mod.fit(Xdata[self.raw_data.outliers!=-1][self.feature_focus])
        
        Xpca = self.pca_mod.transform(Xdata[self.feature_focus])
        if self.principle:
            pass
        else:
            self.principle=Xpca.shape[1]
        
        ## raw PCA
        self.raw_data_pca=pd.DataFrame(Xpca,columns=['p_'+str(i+1) for i in range(self.principle)])
        
        ## PCA component
        ebouli = pd.Series(self.pca_mod.explained_variance_ratio_)
        ebouli_cum = ebouli.cumsum()
        self.pca_component_contribution=pd.DataFrame({'Variance Contribution' :ebouli,
                                                      'Cumulative Variance Contribution':ebouli_cum})
        
        ## Feature Quality with p1 and p2
        coef = np.transpose(self.pca_mod.components_)
        pc_infos = pd.DataFrame(coef, columns=['p_'+str(i+1) for i in range(self.principle)], index=self.feature_focus)
        pc_infos['quality']=np.sqrt(pc_infos['p_1']**2+pc_infos['p_2']**2)
        self.df_features_quality=pc_infos.sort_values('quality',ascending=False)[['quality']]
        self.principle_infos=pc_infos
    
    def get_best_quality_feature(self,correlation_benchmark=0.0,top_benchmark='mean'):
        
        if top_benchmark=='mean':
            self.top_benchmark=self.principle_infos.quality.mean()
        else:
            self.top_benchmark=top_benchmark

        self.correlation_benchmark=correlation_benchmark
        self.Best_Features_Quality=self.principle_infos[self.principle_infos.quality>self.top_benchmark].sort_values('quality',
                                ascending=False).index.tolist()
        self.features_value=self.principle_infos[self.principle_infos.quality>self.top_benchmark]['quality']
        Xdata=self.raw_data[self.feature_focus]
        self.corr_mat=Xdata[self.Best_Features_Quality].corr()**2
        if correlation_benchmark>0:
            self.Best_Features_Quality=clean_high_corr(self.Best_Features_Quality,
                                                       self.corr_mat,
                                                       self.features_value,
                                                       bench=self.correlation_benchmark)
            
    def coc_plot(self,p1="p_1",p2="p_2",rad=0.28,top_quality=10,annot=False):
        circleOfCorrelations(pc_infos1=self.principle_infos,
                             ebouli=self.pca_component_contribution['Variance Contribution'],
                             p1=p1,
                             p2=p2,
                             rad=rad,
                             top_quality=top_quality,
                             annot=annot)
    


#     def clusterization(self,module_cluster_dict={},feature_cluster_use=None):
#         self.module_cluster_dict=module_cluster_dict
#         if feature_cluster_use is None:
#             self.feature_cluster_use=self.Best_Features_Quality
#       else:
#         self.feature_cluster_use=feature_cluster_use
      
#       ## normalize it
#       Xdata0=self.my_raw_data[self.my_raw_data.soft_outlier==0][self.feature_cluster_use]
#       Xdata1=self.my_raw_data[self.feature_cluster_use]
#       Xdata=(Xdata0 - Xdata0.mean()) / Xdata0.std()
#       Xdata_all=(Xdata1 - Xdata0.mean()) / Xdata0.std()
      
#       Xdata=Xdata.fillna(0)
#       Xdata_all=Xdata_all.fillna(0)
      
#       self.Max_Silhouet=0
#       self.Best_Clustering_Algo=""
#       self.Best_Cluster_label=[]
      
#       ## Try many different types of clustering  
#       print('fitting cluster')
#       for cl_model in self.module_cluster_dict.keys():
#         print(cl_model)
#         clf=self.module_cluster_dict[cl_model]
        
#         if Xdata.shape[0]>28000:
#           print("more than 28000 data points")
#           clf.fit(Xdata.sample(28000,random_state=992))
#         else:
#           clf.fit(Xdata)
        
#         try:
#           label_cls=clf.fit_predict(Xdata_all)
#         except:
#           print(cl_model)
#           label_cls=clf.predict(Xdata_all)
#         print('labeling')
#         Silhouet_Val=silhouette_score(Xdata_all, label_cls)
#         if Silhouet_Val>self.Max_Silhouet:
#           self.Max_Silhouet=Silhouet_Val
#           self.Best_Clustering_Algo=clf
#           self.Best_Cluster_label=label_cls
#         print(" ")
      
#       self.my_raw_data['cls']=self.Best_Cluster_label      

#       print("clustering done, Best Algo is ",self.Best_Clustering_Algo, " with Silhouette Value ",self.Max_Silhouet)
      
#   def summary_table(self,title_plot='Nightswatch'):      
#       # Normalize
#       Xdata0=self.my_raw_data[self.my_raw_data.soft_outlier==0][self.feature_cluster_use]
#       Xdata1=self.my_raw_data[self.feature_cluster_use]
#       Xdata=(Xdata0 - Xdata0.mean()) / Xdata0.std()
#       Xdata_all=(Xdata1 - Xdata0.mean()) / Xdata0.std()
#       draw_normalize=pd.DataFrame(Xdata_all,columns=self.feature_cluster_use)
#       draw_normalize['cls']=self.Best_Cluster_label
#       plt.figure(figsize=(10,10))
#       plt.title(title_plot)
#       sns.heatmap(draw_normalize.groupby('cls')[self.feature_cluster_use].mean().transpose(),annot=True,
#                   vmin=-2,vmax=2, cmap=sns.color_palette("BrBG", 7))