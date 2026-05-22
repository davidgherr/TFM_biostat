import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import imblearn as im
from sklearn import metrics,base
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_predict,RepeatedStratifiedKFold as RSKF
from sklearn.calibration import CalibratedClassifierCV,CalibrationDisplay
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
import calzone as cal
import seaborn as sns
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


from tqdm.notebook import tqdm
import matplotlib.ticker as mticker
from matplotlib import gridspec
np.random.seed(777)

class ColSelector(base.ClassifierMixin, base.BaseEstimator):
    """Pensado para el STACK, predice en un modelo con las columnas correspondientes."""
    def __init__(self,model,cols):
        self.model=model
        self.cols=cols
    def fit(self,X,y=None):
        return self
    def predict(self,X):
        return self.model.predict(X[self.cols])
    def predict_proba(self,X):
        return self.model.predict_proba(X[self.cols])
    def __sklearn_is_fitted__(self):
        #Necesario cuando se usa FrozenEstimator, una flag interna de fit.
        return True
def STACK(base1,base2,final,passthrough=False,name=None):
    """Stacking sobre modelos de primer (base1) y segundo trimestre (base2) preentrenados, 
    permitiendo o no pasar covariables al paso final (preentrenado).
    Ojo con la correlación entre las predicciones, coeficientes inestables"""
    S=StackingClassifier(
            estimators=[
                ("T1",ColSelector(base1,base1.feature_names_in_)),
                ("T2",ColSelector(base2,base2.feature_names_in_))
                ],
    final_estimator=FrozenEstimator(final),
    cv="prefit",
    stack_method="predict_proba",
    passthrough=passthrough)
    if name is not None:
        S._report_name=name
    return S



def stack_oof_scores(eval1,name1,y1,eval2,name2,y2,
                     sampler="default",
                     cache_tag_contains=None,
                     score_key="y_score_oof",
                     labels=("T1","T2")):
    """
    Construye el dataframe de scores OOF para STACK.
    """
    res1=eval1.get_cached_cv_result(name1,sampler=sampler,
                              cache_tag_contains=cache_tag_contains)
    res2=eval2.get_cached_cv_result(name2,sampler=sampler,
                              cache_tag_contains=cache_tag_contains)
    s1=pd.Series(res1[score_key],index=y1.index,name=labels[0])
    s2=pd.Series(res2[score_key],index=y2.index,name=labels[1])
    return pd.merge(s1,s2,left_index=True,right_index=True)

    
class ModelReport:
    """Clase genérica para reportar un modelo"""
    def __init__(self,pos_label="claseSdDown",neg_label="claseControl",n_boot=2000,ci_level=0.95,random_state=777):
        self.pos_label=pos_label
        self.neg_label=neg_label
        self.n_boot=n_boot
        self.ci_level=ci_level
        self.random_state=random_state
    #__________ UTILS ___________
    def _name(self,model):
        """Extrae el nombre del estimador""" 
        if hasattr(model,"_report_name"): #Es STACK
            name=model._report_name
        elif hasattr(model,"steps"): #Está en un Pipeline
            name=re.split(r'^(\w+)',model.steps[-1][1].__repr__())[1]
        else: #Es simplemente un modelo
            parts=re.split(r'^(\w+)',model.__repr__())
            if len(parts)>1 and parts[1]:
                name=parts[1]
            else:
                name=type(model).__name__
        return name
    def _binary(self,y):
        """ Pensado para y_true, genera respuesta binaria según clase positiva."""
        return (np.asarray(y)==self.pos_label).astype(int)
    def _proba(self,model, X):
        """Devuelve los scores de la clase positiva"""
        return model.predict_proba(X)[:,1]
    def _sensitivity_at_fpr(self,fpr,tpr,target__fpr=0.05):
        #tpr5
        idx=np.argmin(np.abs(fpr-target__fpr))
        return tpr[idx]
    def _average_precision(self,y_true,y_score):
        if len(np.unique(y_true))<2:
            return 0.0
        return metrics.average_precision_score(self._binary(y_true),y_score)
    def _WBrierSkillScore(self,y_true,y_score,p_ref=None,eps=1e-9):
        """WBSS=1-WBS(model)/WBS(ref)"""
        y_bin=self._binary(y_true)
        p=np.asarray(y_score,dtype=float)
        if p_ref is None:
            #La referencia se considera que estima siempre la frecuencia de la clase positiva.
            ref=np.full_like(y_bin,y_bin.mean(),dtype=float)
        else:
            ref=np.full_like(y_bin,p_ref,dtype=float)
        weights=np.where(y_bin,1/y_bin.mean(),1/(1-y_bin.mean()))
        bs_model=metrics.brier_score_loss(y_bin,p,sample_weight=weights)
        bs_ref=metrics.brier_score_loss(y_bin,ref,sample_weight=weights)
        if bs_ref<eps:
            return 0.0
        return 1-(bs_model/bs_ref)
    def _lr_pos(self,sens,spec): 
        """Likelihood ratio de la clase positiva"""
        return sens/(1-spec) if spec!=1 else np.inf
    def _lr_neg(self,sens,spec):
        """Likelihood ratio de la clase negativa"""
        return (1-sens)/spec if spec!=0 else np.inf
    def _metric_vector(self,y_true,y_pred,y_score):
        """Computa todas las métricas"""
        sens=im.metrics.sensitivity_score(y_true,y_pred,pos_label=self.pos_label)
        spec=im.metrics.specificity_score(y_true,y_pred,pos_label=self.pos_label)
        fpr,tpr,_=metrics.roc_curve(y_true,y_score,pos_label=self.pos_label)
        return {
            "sens": sens,
            "spec": spec,
            "gmean":np.sqrt(sens*spec),
            "tpr5": self._sensitivity_at_fpr(fpr,tpr,0.05),
            "wbss": self._WBrierSkillScore(y_true,y_score),
            "ap": self._average_precision(y_true,y_score),
            "lr+": self._lr_pos(sens,spec),
            "lr-": self._lr_neg(sens,spec)
        }
    def _bootstrap_ci(self,y_true,y_pred,y_score):
        """Intervalos de confianza mediante Bootstrap"""
        y_true=np.asarray(y_true)
        y_pred=np.asarray(y_pred)
        y_score=np.asarray(y_score,dtype=float)
        n=len(y_true)
        alpha=(1-self.ci_level)/2
        rng=np.random.default_rng(self.random_state) #Generador aleatorio con semilla
        samples={k:[] for k in ["sens","spec","gmean","tpr5","wbss","ap","lr+","lr-"]}
        def _rank_percentile(arr,q):
            #Saca el percentil q de una muestra (arr)
            return np.percentile(arr,100*q,method="lower")
        for _ in range(self.n_boot):
            idx=rng.integers(0,n,n) #Variación con repetición
            y_b=y_true[idx]
            if len(np.unique(y_b))<2:
                continue #Muestra degenerada
            vals=self._metric_vector(y_b,y_pred[idx],y_score[idx])
            for key,val in vals.items():
                samples[key].append(val)
        ci={}
        for key,vals in samples.items():
            arr=np.asarray(vals,dtype=float)
            arr=arr[~np.isnan(arr)]
            if arr.size==0: #Algo ha ido mal y son todos NaN
                ci[key]=(np.nan,np.nan)
            else:
                ci[key]=(
                    _rank_percentile(arr,alpha),
                    _rank_percentile(arr,1-alpha)
                )
        return ci
    def _fmt_value(self,val,ci=None,digits=3):
        """Pensado para el print, da formato a los valores con o sin IC"""
        def _fmt_bound(x):
            if np.isnan(x):
                return "nan"
            if np.isinf(x):
                return "inf"
            return f"{x:.{digits}f}"
        
        text=_fmt_bound(val)
        if ci is None:
            return text
        low,high=ci
        return f"{text} [{_fmt_bound(low)}, {_fmt_bound(high)}]"
    #______  METRICAS_________
    def compute_metrics(self, model, X, y,threshold=0.5):
        y_score=self._proba(model,X)
        y_pred=np.where(y_score<=threshold,self.neg_label, self.pos_label)
        metrics_dict=self._metric_vector(y,y_pred,y_score)
        y_bin=self._binary(y)
        return {
            **metrics_dict,
            "ci": self._bootstrap_ci(y,y_pred,y_score),
            "sup_pos": int(y_bin.sum()),
            "sup_neg": int((1-y_bin).sum()),
            "prev": float(y_bin.mean())
        }
    def print_report(self,model,X,y,threshold=0.5):
        name=self._name(model)
        m=self.compute_metrics(model,X,y,threshold)
        print(f'\033[1m{name}\033[0m\n'+'_'*85)
        print(f'sup_neg: {m["sup_neg"]}  sup_pos: {m["sup_pos"]}  prev: {m["prev"]:.3f}')
        print(f'sens: {self._fmt_value(m["sens"],m["ci"]["sens"])}')
        print(f'spec: {self._fmt_value(m["spec"],m["ci"]["spec"])}')
        print(f'gmean: {self._fmt_value(m["gmean"],m["ci"]["gmean"])}')
        print(f'tpr5: {self._fmt_value(m["tpr5"],m["ci"]["tpr5"])}')
        print(f'WBSS:  {self._fmt_value(m["wbss"],m["ci"]["wbss"])}')
        print(f'AP:   {self._fmt_value(m["ap"],m["ci"]["ap"])}')
        print(f'LR+:  {self._fmt_value(m["lr+"],m["ci"]["lr+"])}')
        print(f'LR-:  {self._fmt_value(m["lr-"],m["ci"]["lr-"])}\n')
    #______________ PLOTS ________________

    def plot_confusion(self, model, X, y,threshold=0.5 ,ax=None):
        """Matriz de confusión normalizada por valores reales"""
        name=self._name(model)
        y_score=self._proba(model,X)
        y_pred=np.where(y_score<=threshold,self.neg_label, self.pos_label)
        metrics.ConfusionMatrixDisplay.from_predictions(
            y,y_pred,
            normalize="true",
            cmap="gray",
            colorbar=False,
            ax=ax
        )
        ax.set_title(name) if ax else plt.title(name)

    # _____________ PIPELINE ________________-
    def evaluate(self,models, X,y,thresholds):
        if not isinstance(models,list):
            models=[models] 
            thresholds=[thresholds]
        self.plot_curves(models,X,y) #Curvas
        #Reportes
        fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),5))
        for ax,model,threshold in zip(np.atleast_1d(axes),models,thresholds):
            self.plot_confusion(model,X,y,ax=ax,threshold=threshold)
            self.print_report(model,X,y,threshold=threshold)
        plt.show()
    
    def plot_curves(self,models,X,y):
        """Curvas ROC, PR y DET usando from_estimator"""
        fig,ax=plt.subplots(1,3,figsize=(20,5))
        for model in models:
            metrics.DetCurveDisplay.from_estimator(model,X,y,ax=ax[0],pos_label=self.pos_label,name=self._name(model))
            metrics.PrecisionRecallDisplay.from_estimator(model,X,y,ax=ax[1],pos_label=self.pos_label,name=self._name(model))
            metrics.RocCurveDisplay.from_estimator(model,X,y,ax=ax[2],pos_label=self.pos_label,name=self._name(model))
        ax[0].set_title("Curva DET")
        ax[1].set_title("Curva Precision-Recall")
        ax[2].set_title("Curva ROC")
        plt.show()   

class CVEvaluator:
    """
    Evaluador con validación cruzada estratificada.
    Recorre los folds una vez por modelo y cachea:
        y_score_oof: scores out-of-fold sobre todo el train (calibradas)
        y_score_oof_raw: scores OOF sin calibrar
        y_pred_oof: predicción OOF sobre un umbral que optimiza la métrica tpr5
        tprs_folds: TPR interpolada por fold (para varianza)
        fold_metrics: Métricas por fold 
        stats: media y sd de las métricas
        pipeline: con calibrador
        pipeline_raw: sin calibrador
        Grid_Res: Resultado de las métricas en optuna
        opt_metrics: Métricas utilizadas para optimizar
        cache_tag:pseudo identificador con las características de la llamada
        pareto_strategy
        pareto_fronts
        
    compare_models() genera una salida numérica y gráfica.
    """

    def __init__(self,cv,pos_label="claseSdDown",sampler=None,sampling_strategy=0.3,random_state=777):
        self.cv=cv
        self.pos_label=pos_label
        self.sampling_strategy=sampling_strategy
        self.random_state=random_state
        self.default_sampler=sampler
        self._cache={} #{(model_name,sampler_id,tags):CVResult}
    # ___________helpers de pipeline  _____________
    def get_cached_cv_result(self,model_name,sampler="default",cache_tag_contains=None):
        """
        Recupera resultados de la caché de `CVEvaluator` filtrando por nombre de
        modelo y, opcionalmente, por un fragmento del `cache_tag`.
        """
        active_sampler=self.default_sampler if sampler=="default" else sampler
        sampler_key=self._sampler_key(active_sampler)
        matches=[
            (key,result)
            for key,result in self._cache.items()
            if key[0]==model_name and key[1]==sampler_key
        ] #Todos los que comparten nombre y sampler, generalmente sólo hay 1
        if cache_tag_contains is not None:
            matches=[
                (key,result)
                for key,result in matches
                if cache_tag_contains in key[2]
            ]
        if len(matches)!=1:
            tags=[key[2] for key,_ in matches]
            raise ValueError(
                f"No hay resultado único para '{model_name}'. "
                f"Tags encontrados: {tags}"
            )
        return matches[0][1]
    def _build_pipeline(self,estimator,sampler="default"):
        """Construye un pipe con el sampler si hay, junto al clasificador"""  
        if sampler == "default":
            return self._build_pipeline(estimator,self.default_sampler)
        if sampler is None:
            return base.clone(estimator)
        return ImbPipeline([
            ("sampler", base.clone(sampler)),
            ("classifier",base.clone(estimator))
        ])
    def _prefix_grid(self,param_grid,sampler):
        """Añade prefijo a param_grid para referirse al clasificador"""
        if sampler is None or param_grid is None:
            return param_grid
        return {f"classifier__{k}":v for k,v in param_grid.items()}
    def _pauc(self,y_true,y_score,fpr_target=0.05):
        #Calcular el AUC parcial 
        if len(np.unique(y_true))<2:
            return 0.5
        return metrics.roc_auc_score(y_true,y_score,max_fpr=fpr_target)
    def _tpr_at_fpr(self,y_true,y_score,fpr_target=0.05):
        #tpr5 interpolado
        if len(np.unique(y_true))<2:
            return 0.0
        fpr,tpr,_=metrics.roc_curve(y_true,y_score,pos_label=self.pos_label)
        if np.max(fpr)<fpr_target:
            return tpr[-1]
        return np.interp(fpr_target,fpr,tpr)
    def _roc_auc(self,y_true,y_score):
        if len(np.unique(y_true))<2:
            return 0.5
        return metrics.roc_auc_score(y_true,y_score)
    def _average_precision(self,y_true,y_score):
        if len(np.unique(y_true))<2:
            return 0.0
        y_bin=(np.asarray(y_true)==self.pos_label).astype(int)
        return metrics.average_precision_score(y_bin,y_score)
    def _weighted_Brier_Score(self,y_true,y_score):
        """
        Brier Score ponderado por clase.

        Cada clase recibe el mismo peso agregado. Si el fold contiene una única
        clase, devuelve `np.inf` para penalizar ese trial en un criterio a minimizar.
        """
        y_bin=(np.asarray(y_true)==self.pos_label).astype(int)
        pi=y_bin.mean()
        if pi in [0,1]:
            return np.inf
        p=np.asarray(y_score,dtype=float)
        weights=np.where(y_bin==1,1/pi,1/(1-pi))
        return metrics.brier_score_loss(y_bin,p,sample_weight=weights)/2
    def _focal_loss(self,y_true,y_score,alpha=0.25,gamma=2.0,eps=1e-9):
        """
        Binary form of focal loss.
          FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        """
        y_bin=(np.asarray(y_true)==self.pos_label).astype(int)
        p=np.clip(np.asarray(y_score,dtype=float),eps,1-eps)
        pt=np.where(y_bin==1,p,1-p)
        at=np.where(y_bin==1,alpha,1-alpha)
        loss=-at*((1-pt)**gamma)*np.log(pt)
        return loss.mean()
    def _BrierSkillScore(self,y_true,y_score,p_ref=None,eps=1e-9):
        """WBSS=1-WBS(model)/WBS(ref)"""
        y_bin=(np.asarray(y_true)==self.pos_label).astype(int)
        p=np.asarray(y_score,dtype=float)
        if p_ref is None:
            ref=np.full_like(y_bin,y_bin.mean(),dtype=float)
        else:
            ref=np.full_like(y_bin,p_ref,dtype=float)
        bs_model=self._weighted_Brier_Score(y_true,p)
        bs_ref=self._weighted_Brier_Score(y_true,ref)
        if bs_ref<eps:
            return 0.0
        return 1- (bs_model/bs_ref)
    def _opt_metric_catalog(self):
        """Catálogo corto para tuning, con posibilidad de ampliar desde fuera."""
        return {
            "tpr5": {"func": self._tpr_at_fpr, "direction": "maximize"},
            "pauc5": {"func": self._pauc, "direction": "maximize"},
            "roc_auc": {"func": self._roc_auc, "direction": "maximize"},
            "ap": {"func": self._average_precision, "direction": "maximize"},
            "WBS": {"func": self._weighted_Brier_Score, "direction": "minimize"},
            "focal_loss":{"func":self._focal_loss,"direction":"minimize"},
            "WBSS":{"func":self._BrierSkillScore,"direction":"maximize"}
        }
    def _resolve_opt_metrics(self,opt_metrics=None):
        """
        Normaliza opt_metrics a una lista de specs:
            {"name": ..., "func": ..., "direction": ...}

        Formatos válidos:
            None -> "tpr5"
            "roc_auc"
            ["tpr5","ap"]
            {"tpr5":[callable,"maximize"]}
            {"tpr5":{"func": callable, "direction":"maximize"}}
        """
        catalog=self._opt_metric_catalog()
        if opt_metrics is None:
            opt_metrics="tpr5"
        if isinstance(opt_metrics,str):
            opt_metrics=[opt_metrics]
        if isinstance(opt_metrics,(list,tuple)):
            resolved=[]
            for item in opt_metrics:
                if not isinstance(item,str):
                    raise TypeError("Si opt_metrics es lista, cada elemento debe ser un string")
                if item not in catalog:
                    raise ValueError(f"Métrica no en catalogo: {item}")
                resolved.append({"name": item} | catalog[item])
            return resolved
        if isinstance(opt_metrics,dict):
            resolved=[]
            for name,spec in opt_metrics.items():
                if isinstance(spec,(list,tuple)) and len(spec)==2:
                    func,direction=spec
                elif isinstance(spec,dict):
                    func=spec["func"]
                    direction=spec["direction"]
                else:
                    raise TypeError(
                        "Cada entrada de opt_metrics debe ser [callable, direction] "
                        "o {'func': callable, 'direction': direction}."
                    )
                if direction not in {"maximize","minimize"}:
                    raise ValueError(f"Dirección inválida para {name}: {direction}")
                resolved.append({"name": name, "func": func, "direction": direction})
            return resolved
        raise TypeError("opt_metrics debe ser None, str, list/tuple o dict.")
    def _compute_opt_metric_values(self,opt_metric_specs,y_true,y_score):
        """Evalúa las métricas de optimización sobre un vector de scores."""
        return [spec["func"](y_true,y_score) for spec in opt_metric_specs]
    def _threshold_at_fpr(self,y_true,y_score,fpr_target=0.05):
        """
        Obtiene el umbral (u5) asociado a un FPR objetivo. 
        Se ajusta en train y se aplica a test, excepto si el conjunto es degenerado
        """
        if len(np.unique(y_true))<2:
            return 1.0
        fpr,_,thresholds=metrics.roc_curve(y_true,y_score,pos_label=self.pos_label)
        return np.interp(fpr_target,fpr,thresholds)
    def _aligned_matrix(self,value_matrix,opt_metric_specs):
        """
        Reorienta una matriz de métricas para que todas apunten a maximizar.
        """
        signed=np.asarray(value_matrix,dtype=float).copy()
        for idx,spec in enumerate(opt_metric_specs):
            if spec["direction"]=="minimize":
                signed[:,idx]*=-1
        return signed
    def _select_best_index(self,value_matrix,opt_metric_specs,
                           pareto_strategy="lexicographic",pareto_weights=None):
        """
        Selecciona el mejor candidato entre varios vectores de métricas.

        Se usa tanto sobre trials de Optuna como sobre modelos ya resumidos en
        `compare_models()`, manteniendo el mismo criterio multiobjetivo.
        """
        signed=self._aligned_matrix(value_matrix,opt_metric_specs)
        if signed.ndim!=2 or signed.shape[0]==0:
            raise ValueError("value_matrix debe tener forma (n_candidatos, n_metricas).")
        if pareto_strategy=="lexicographic":
            return max(range(len(signed)),key=lambda i: tuple(signed[i]))
        if pareto_strategy=="weighted_sum":
            weights=np.asarray(pareto_weights or [1.0]*len(opt_metric_specs),dtype=float)
            if len(weights)!=len(opt_metric_specs):
                raise ValueError("pareto_weights debe tener la misma longitud que opt_metrics.")
            return int(np.argmax(signed @ weights))
        if pareto_strategy =="topsis":
            # TOPSIS clásico: normalización vectorial,
            # ponderación, ideal/anti-ideal y cercanía relativa.
            weights=np.asarray(pareto_weights or [1.0]*len(opt_metric_specs),dtype=float)
            if len(weights)!=len(opt_metric_specs):
                raise ValueError("pareto_weights debe tener la misma longitud que opt_metrics.")
            weight_sum=weights.sum()
            if weight_sum<=0:
                raise ValueError("pareto_weights deben ser no negativos.")
            weights=weights/weight_sum
            norms=np.linalg.norm(signed,axis=0)
            norms=np.where(norms==0,1.0,norms)
            normalized=signed/norms
            weighted=normalized*weights
            ideal=weighted.max(axis=0)
            anti_ideal=weighted.min(axis=0)
            d_pos=np.linalg.norm(weighted-ideal,axis=1)
            d_neg=np.linalg.norm(weighted-anti_ideal,axis=1)
            closeness=d_neg/(d_pos+d_neg+1e-12)
            return int(np.argmax(closeness))
        raise ValueError(
            "pareto_strategy debe ser 'lexicographic', 'weighted_sum', "
            "'topsis'."
        )
    def _cache_tag(self,param_grid,calibrate,calibration_method,opt_metric_names,
                   pareto_strategy="lexicographic"):
        """
        Construye una etiqueta para la caché.

        La clave completa queda como (name, sampler, tag), donde `tag` resume: 
        tuning, calibración y criterio de optimización.
        """
        parts=["grid" if param_grid else "nogrid"]
        if calibrate:
            parts.append(f"cal_{calibration_method}")
        else:
            parts.append("raw")
        parts.append("opt_" + "-".join(opt_metric_names))
        if len(opt_metric_names)>1:
            parts.append(f"pareto_{pareto_strategy}")
        return "__".join(parts)
    def _select_pareto_trial(self,trials,opt_metric_specs,pareto_strategy="lexicographic",
                             pareto_weights=None):
        """
        Elige un único trial a partir del frente de Pareto.

        Estrategias soportadas:
            - `lexicographic`: prioriza las métricas en el orden declarado.
            - `weighted_sum`: suma ponderada tras reorientar los signos.
            - `topsis`: TOPSIS clásico.

        Notas:
            - `trials` debe ser `study.best_trials`.
            - Para `weighted_sum`, si `pareto_weights` es `None`, se usan pesos uniformes.
        """
        if not trials:
            raise ValueError("No hay trials en el frente de Pareto para seleccionar.")
        values=np.array([trial.values for trial in trials],dtype=float)
        best_idx=self._select_best_index(values,opt_metric_specs,
                                         pareto_strategy=pareto_strategy,
                                         pareto_weights=pareto_weights)
        return trials[best_idx]
        
    def _sampler_key(self,sampler):
        #Nombre del sampler 
        if sampler=="default":
            sampler=self.default_sampler
        if sampler is None:
            return "none"
        
        return f"{type(sampler).__name__}"
    # ________________RECORRIDO_______________
    def _run_cv(self,name,model,X,y,param_grid=None,sampler="default",
               calibrate=True,calibration_method="sigmoid",
               opt_metrics=None,pareto_strategy="lexicographic",
               pareto_weights=None):
        """
        Recorre los folds una vez y almacena todo en self._cache.

        opt_metrics controla la optimización interna de Optuna. Puede ser:
            None -> usa "tpr5"
            "roc_auc"
            ["tpr5","ap"]
            {"tpr5":[callable,"maximize"]}

        Si hay varias métricas, `pareto_strategy` resuelve el frente de Pareto
        en un único trial para el refit final y para cada fold externo.

        Nested CV estima generalización; después hace un refit final sobre todo X,y.
        """
        def _prepare_params(base_pipe,trial):
            params=param_grid(trial)
            if hasattr(base_pipe,"steps"):
                params={f"classifier__{k}":v for k,v in params.items()}
            return params

        def _build_study(study_name):
            study_kwargs={"study_name": study_name}
            if len(opt_metric_specs)>1:
                study_kwargs["directions"]=opt_metric_dirs
                study_kwargs["sampler"]=optuna.samplers.NSGAIISampler(seed=self.random_state)
            else:
                study_kwargs["direction"]=opt_metric_dirs[0]
                study_kwargs["sampler"]=optuna.samplers.TPESampler(seed=self.random_state)
            study=optuna.create_study(**study_kwargs)
            study.set_metric_names(opt_metric_names)
            return study

        def _best_trial_from_study(study):
            if len(opt_metric_specs)==1:
                return study.best_trial
            return self._select_pareto_trial(
                study.best_trials,
                opt_metric_specs,
                pareto_strategy=pareto_strategy,
                pareto_weights=pareto_weights
            )

        def _best_params_from_study(study,base_pipe):
            best_trial=_best_trial_from_study(study)
            best_params=best_trial.params
            if hasattr(base_pipe,"steps"):
                best_params={f"classifier__{k}":v for k,v in best_params.items()}
            return best_params,best_trial

        neg_label=["claseSdDown","claseControl"]
        neg_label.remove(self.pos_label)
        opt_metric_specs=self._resolve_opt_metrics(opt_metrics) #{name:,func:,direction:}
        opt_metric_names=[spec["name"] for spec in opt_metric_specs]
        opt_metric_dirs=[spec["direction"] for spec in opt_metric_specs]
        cache_tag=self._cache_tag(param_grid,calibrate,calibration_method,
                                  opt_metric_names,pareto_strategy=pareto_strategy)
        cache_key=(name,self._sampler_key(sampler),cache_tag)
        if cache_key in self._cache:
            return self._cache[cache_key]

        mean_fpr=np.linspace(0,1,100)
        y_score_oof=np.zeros(len(y))
        y_score_oof_raw=np.zeros(len(y))
        y_pred_oof=np.empty(len(y),dtype=object)
        tprs_folds=[]
        fold_metrics=[]
        cv_Grid_results=[]
        pareto_fronts=[]

        for train_idx,test_idx in self.cv.split(X,y):
            Xtr,Xte=X.iloc[train_idx],X.iloc[test_idx]
            ytr,yte=y.iloc[train_idx],y.iloc[test_idx]
            pipe=self._build_pipeline(base.clone(model),sampler)

            if param_grid:
                def objective(trial):
                    trial_pipe=base.clone(pipe)
                    trial_pipe.set_params(**_prepare_params(pipe,trial))
                    inner_cv=RSKF(n_splits=3,n_repeats=1,random_state=self.random_state)
                    oof_scores=cross_val_predict(trial_pipe,Xtr,ytr,
                                                 cv=inner_cv,
                                                 method="predict_proba")[:,1]
                    values=self._compute_opt_metric_values(opt_metric_specs,ytr,oof_scores)
                    return values[0] if len(values)==1 else tuple(values)

                study=_build_study(f"Outer fold {test_idx[0]}")
                study.optimize(objective,n_trials=50,show_progress_bar=False,n_jobs=-1)
                best_params,best_trial=_best_params_from_study(study,pipe)
                pipe.set_params(**best_params)
                cv_Grid_results.append(study.trials_dataframe())
                pareto_fronts.append({
                    "scope": "outer_fold",
                    "fold_start_idx": test_idx[0],
                    "selected_trial_number": int(best_trial.number),
                    "selected_values": tuple(best_trial.values) if best_trial.values is not None else None,
                    "front": [
                        {
                            "trial_number": int(trial.number),
                            "values": tuple(trial.values) if trial.values is not None else None,
                            "params": trial.params
                        }
                        for trial in study.best_trials
                    ] if len(opt_metric_specs)>1 else []
                })

            pipe.fit(Xtr,ytr)
            raw_scores=pipe.predict_proba(Xte)[:,1]
            y_score_oof_raw[test_idx]=raw_scores

            if calibrate:
                cal=CalibratedClassifierCV(
                    pipe,
                    method=calibration_method,
                    cv=RSKF(n_splits=3,n_repeats=5,random_state=self.random_state)
                )
                cal.fit(Xtr,ytr)
                train_scores=cal.predict_proba(Xtr)[:,1]
                scores=cal.predict_proba(Xte)[:,1]
            else:
                train_scores=pipe.predict_proba(Xtr)[:,1]
                scores=raw_scores

            # El umbral se aprende en train y se aplica al fold de test.
            u5=self._threshold_at_fpr(ytr,train_scores,0.05)
            preds=np.where(scores>=u5,self.pos_label,neg_label[0]) #

            y_score_oof[test_idx]=scores
            y_pred_oof[test_idx]=preds

            sens=im.metrics.sensitivity_score(yte,preds,pos_label=self.pos_label)
            spec=im.metrics.specificity_score(yte,preds,pos_label=self.pos_label)
            fold_row={
                "sensitivity": sens,
                "specificity": spec,
                "balanced_accuracy": (sens+spec)/2,
                "gmean": im.metrics.geometric_mean_score(yte,preds),
                "tpr5": self._tpr_at_fpr(yte,scores),
            }
            # Añadimos también las métricas de optimización.
            fold_row.update(dict(zip(
                opt_metric_names,
                self._compute_opt_metric_values(opt_metric_specs,yte,scores)
            )))
            fold_metrics.append(fold_row)

            fpr_curve,tpr_curve,_=metrics.roc_curve(yte,scores,pos_label=self.pos_label)
            tprs_folds.append(np.interp(mean_fpr,fpr_curve,tpr_curve))

        final_pipe=self._build_pipeline(base.clone(model),sampler)
        if param_grid:
            search_pipe=self._build_pipeline(base.clone(model),sampler)

            def objective(trial):
                trial_pipe=base.clone(search_pipe)
                trial_pipe.set_params(**_prepare_params(search_pipe,trial))
                inner_cv=RSKF(n_splits=3,n_repeats=1,random_state=self.random_state)
                oof_scores=cross_val_predict(trial_pipe,X,y,
                                             cv=inner_cv,
                                             method="predict_proba")[:,1]
                values=self._compute_opt_metric_values(opt_metric_specs,y,oof_scores)
                return values[0] if len(values)==1 else tuple(values)

            study=_build_study(f"{name} final refit")
            study.optimize(objective,n_trials=50,show_progress_bar=False,n_jobs=-1)
            best_params,best_trial=_best_params_from_study(study,final_pipe)
            final_pipe.set_params(**best_params)
            cv_Grid_results.append(study.trials_dataframe())
            pareto_fronts.append({
                "scope": "final_refit",
                "selected_trial_number": int(best_trial.number),
                "selected_values": tuple(best_trial.values) if best_trial.values is not None else None,
                "front": [
                    {
                        "trial_number": int(trial.number),
                        "values": tuple(trial.values) if trial.values is not None else None,
                        "params": trial.params
                    }
                    for trial in study.best_trials
                ] if len(opt_metric_specs)>1 else []
            })

        if calibrate:
            final_calibrated=CalibratedClassifierCV(
                final_pipe,
                cv=RSKF(n_splits=3,n_repeats=5,random_state=self.random_state),
                method=calibration_method
            )
            final_calibrated.fit(X,y)
        else:
            final_calibrated=None
        final_pipe.fit(X,y)

        df_folds=pd.DataFrame(fold_metrics)
        result={
            "y_score_oof":y_score_oof,
            "y_score_oof_raw":y_score_oof_raw,
            "y_pred_oof":y_pred_oof,
            "tprs_folds":tprs_folds,
            "mean_fpr":mean_fpr,
            "fold_metrics":df_folds,
            "stats":df_folds.mean().to_dict() | df_folds.std().add_suffix("_std").to_dict(),
            "pipeline":final_calibrated if calibrate else final_pipe,
            "pipeline_raw":final_pipe,
            "Grid_Res":cv_Grid_results,
            "opt_metrics":opt_metric_names,
            "cache_tag":cache_tag,
            "pareto_strategy":pareto_strategy,
            "pareto_fronts":pareto_fronts,
        }
        self._cache[cache_key]=result
        return result

    # ___________API_________________
    def evaluate_model(self,model,X,y,param_grid=None,sampler="default",name=None,
                       opt_metrics=None,pareto_strategy="lexicographic",
                       pareto_weights=None):
        """
        Ejecuta o recupera caché el recorrido CV para un modelo.
        Devuelve `(pipeline_final, stats_dict)`.

        Si `opt_metrics` define varias métricas, `pareto_strategy` controla cómo
        se elige un único trial del frente de Pareto para el refit final.
        Recomendación: usar `pareto_strategy="topsis"`.
        """
        name=name or re.split(r"^(\w+)",model.__repr__())[1]
        result=self._run_cv(name,model,X,y,param_grid,sampler,
                            opt_metrics=opt_metrics,
                            pareto_strategy=pareto_strategy,
                            pareto_weights=pareto_weights)
        return result["pipeline"],result["stats"]
    
    #  _______________ ROC Out Of Fold____________
    def roc_oof(self,model,X,y,sampler="default",name=None):
        #Devuelve (fpr,tpr,auc) usando scores OOF cacheados
        name=name or re.split(r"^(\w+)",model.__repr__())[1]
        result=self._run_cv(name,model,X,y,None,sampler)
        fpr,tpr,_=metrics.roc_curve(y,result["y_score_oof"],
                                   pos_label=self.pos_label)
        return fpr,tpr,metrics.auc(fpr,tpr)

    # ________________ Correción por prevalencia real_______-
    @staticmethod
    def prior_correction(p,pi_train,pi_real=1/800):
        """
        Corrige las probabilidades del modelo desde la prevalencia de entrenamiento (pi_train, distorsionada por sampler)
        hacia la prevalencia real (pi_real)
        Derivación:
            El modelo aprende P(y=SD|x,pi_train). Para obtener P(y=SD|x,pi_real) se usa el ratio de verosimilitud (LR) que es independiente
            de la prevalencia. LR(x)=P(x|y=SD)/P(x|y=control) = [p/(1-p)]*[(1-pi_train)/pi_train]
            Y luego se aplica Bayes con pi_real:
            p_corr=LR*pi_real/(LR*pi_real+(1-pi_real))
        p: array de probabilidades del modelo
        pi_real: prevalencia real
        pi_train: prevalencia durante el entrenamiento, con sampler: sampling_strategy
        """
        p=np.asarray(p,dtype=float)
        p=np.clip(p,1e-9,1-1e-9)
        lr=(p/(1-p))*((1-pi_train)/pi_train)
        return lr*pi_real/(lr*pi_real+(1-pi_real))
    # ________________diagnóstico y corrección de calibración ____________--
    def plot_calibration(self,models,X,y,
                        pi_real=None,
                        pi_train=None,
                        sampler="default",
                        grids=None,
                        opt_metrics=None,
                        pareto_strategy="lexicographic",
                        pareto_weights=None,
                        n_bins=10,
                        figsize=(6,5)):
        """
        Para cada modelo muestra:
            - Curva de calibración con scores OOF sin calibrar
            - Mapeo de calibración
            - KDE por clase y LR+
        """
        active_sampler=self.default_sampler if sampler=="default" else sampler
        if pi_train is None:
            if active_sampler is None:
                raise Exception("Frecuencia de la clase no especificada (pi_train)")
            pi_train=self.sampling_strategy ###
        palette=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        y_bin=np.array(y==self.pos_label)*1
        for i,(name,model) in enumerate(models.items()):
            grid=grids.get(name) if grids else None
            #recuperar caché o ejecutar
            res=self._run_cv(name,model,X,y,grid,sampler,calibrate=True,
                             opt_metrics=opt_metrics,
                             pareto_strategy=pareto_strategy,
                             pareto_weights=pareto_weights)
            fig,axes=plt.subplots(1,3,figsize=(figsize[0]*3,figsize[1]))
            y_pred=np.array([res["y_score_oof"],res["y_score_oof"]]).transpose()
            z,p=cal.metrics.spiegelhalter_z_test(y_bin,y_pred)
            fig.suptitle(f"Calibración OOF - {name}\n SpH={z:.3f}"+("*" if p<0.05 else ""), 
                         fontsize=12)
            #Sin calibrar
            CalibrationDisplay.from_predictions(y,res["y_score_oof_raw"],
                                               n_bins=n_bins,ax=axes[0],
                                               name="Sin calibrar",color=palette[0],
                                               pos_label=self.pos_label)
            axes[0].set_title("Sin calibrar (raw)")
            #2. Mapeo de Calibrado 
            axes[1].set_title("Mapeo")
            gs=axes[1].get_subplotspec().subgridspec(2,2,
                                        height_ratios=(0.1,0.9),
                                        width_ratios=(0.9,0.1),
                                        hspace=0.05,
                                        wspace=0.05)
            ax_scatter=axes[1].figure.add_subplot(gs[1,0])
            ax_box_y=axes[1].figure.add_subplot(gs[1,1],sharey=ax_scatter)
            ax_box_x=axes[1].figure.add_subplot(gs[0,0],sharex=ax_scatter)
            axes[1].set_visible(False)
            ax_scatter.scatter(res["y_score_oof_raw"],
                               res["y_score_oof"],
                               alpha=0.3,c=y_bin,cmap="viridis")
            ax_scatter.plot([0,1],[0,1],'--')
            ax_scatter.set_xlabel("Uncalibrated Probability (Raw)")
            ax_scatter.set_ylabel("Calibrated Probability")
            sns.boxplot(y=res["y_score_oof"],
                        ax=ax_box_y,hue=y_bin,palette=["purple","yellow"])
            ax_box_y.axis("off")
            sns.boxplot(x=res["y_score_oof_raw"],
                        ax=ax_box_x,hue=y_bin,palette=["purple","yellow"])
            ax_box_x.axis("off")
            
            #3. KDE
            axes[2].set_title("Distribución KDE calibradas")
            sns.kdeplot(x=res["y_score_oof"],
                        hue=y_bin,common_norm=False,fill=True,ax=axes[2],
                       legend=True,bw_adjust=0.5,clip=(0,1),cut=0)
            sns.rugplot(x=res["y_score_oof"][y_bin==1], ax=axes[2], color="orange")
            b=np.sort(res["y_score_oof"])
            axes[2].plot(b,np.log((b/(1-b))*((1-pi_train)/pi_train)),color="black",label="log LR+")
            axes[2].set_xlim(0,1)
            axes[2].set_xlabel("Calibrated Probability")
            
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(lw=0.3,alpha=0.5)
            plt.show()
            
            
        

    # _________________ Comparar VARIOS MODELOS ___________________

    def compare_models(self,models,X,y,sampler="default",grids=None,
                       select_by="auto",opt_metrics=None,
                       pareto_strategy="lexicographic",pareto_weights=None,
                       figsize=(22,5)):
        """
        Para cada modelo: 
            Ejecuta _run_cv()
            Genera tabla media +- sd
            Curvas DET,PR,ROC
            Marca el mejor modelo
        Devuelve (df_stats,fitted_models,best_name)

        `select_by="auto"` hace que la selección final siga el mismo criterio
        que el tuning:
            - una sola métrica -> esa métrica
            - varias métricas -> mismo `pareto_strategy`
        Para multiobjetivo, `pareto_strategy="topsis"` es la opción recomendada.
        """
        #1. Recorrido CV
        opt_metric_specs=self._resolve_opt_metrics(opt_metrics)
        opt_metric_names=[spec["name"] for spec in opt_metric_specs]
        results={}
        for name,model in tqdm(models.items(),desc="Models"):
            grid=grids.get(name) if grids else None
            result=self._run_cv(name,model,X,y,grid,sampler,
                                opt_metrics=opt_metrics,
                                pareto_strategy=pareto_strategy,
                                pareto_weights=pareto_weights)
            results[name]=result
        #2. Selección del mejor
        if select_by=="auto":
            if len(opt_metric_specs)==1:
                select_by=opt_metric_names[0]
                metric_direction=opt_metric_specs[0]["direction"]
                scores_sel={n:r["stats"].get(select_by,0) for n,r in results.items()}
                best_name=(max if metric_direction=="maximize" else min)(scores_sel,key=scores_sel.get)
                ranking_label=select_by
            else:
                value_matrix=np.array([
                    [results[name]["stats"].get(metric_name,np.nan) for metric_name in opt_metric_names]
                    for name in models
                ],dtype=float)
                best_idx=self._select_best_index(
                    value_matrix,
                    opt_metric_specs,
                    pareto_strategy=pareto_strategy,
                    pareto_weights=pareto_weights
                )
                best_name=list(models.keys())[best_idx]
                scores_sel={name: tuple(results[name]["stats"].get(metric_name,np.nan)
                                        for metric_name in opt_metric_names)
                           for name in models}
                ranking_label=f"pareto:{pareto_strategy} -> {', '.join(opt_metric_names)}"
        else:
            scores_sel={n:r["stats"].get(select_by,0) for n,r in results.items()}
            metric_direction="maximize"
            for spec in opt_metric_specs:
                if spec["name"]==select_by:
                    metric_direction=spec["direction"]
                    break
            best_name=(max if metric_direction=="maximize" else min)(scores_sel,key=scores_sel.get)
            ranking_label=select_by
        #3. Estilos
        palette=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color={n:palette[i%len(palette)] for i,n in enumerate(models)}
        def _lw(n): return 2.5 if n==best_name else 1.2
        def _ls(n): return "-" if n==best_name else "--"
        def _star(n): return chr(171) if n==best_name else ""
        fig, axes=plt.subplots(1,4,figsize=figsize)
        mean_fpr=np.linspace(0,1,100)
        for name,res in results.items():
            y_score=res["y_score_oof"]
            y_pred=res["y_pred_oof"]
            label= name+_star(name)
            metrics.DetCurveDisplay.from_predictions(y,y_score,pos_label=self.pos_label,
                                                    name=label,ax=axes[0],
                                                    lw=_lw(name),linestyle=_ls(name),color=color[name])
            metrics.PrecisionRecallDisplay.from_predictions(y,y_score,pos_label=self.pos_label,
                                                    name=label,ax=axes[1],
                                                    lw=_lw(name),linestyle=_ls(name),color=color[name])
            metrics.RocCurveDisplay.from_predictions(y,y_score,pos_label=self.pos_label,
                                                    name=label,ax=axes[2],
                                                    lw=_lw(name),linestyle=_ls(name),color=color[name])
            mean_tpr=np.mean(res["tprs_folds"],axis=0)
            std_tpr=np.std(res["tprs_folds"],axis=0)
            axes[3].plot(mean_fpr,mean_tpr,lw=_lw(name),color=color[name],label=label)
            axes[3].fill_between(mean_fpr,
                              np.maximum(mean_tpr-std_tpr,0),
                              np.minimum(mean_tpr+std_tpr,1),
                              color=color[name],alpha=0.12)
        axes[0].set_title("DET (OOF)")
        axes[1].set_title("Precision-Recall (OOF)")
        axes[2].set_title("ROC (OOF)")
        axes[3].plot([0,1],[0,1],"k:",lw=0.8)
        axes[3].set(xlabel="FPR",ylabel="TPR",title="ROC por folds")
        axes[3].legend(fontsize=8)
        axes[3].grid(lw=0.3,alpha=0.5)
        plt.suptitle(
            f"Comparación OOF - mejor:{best_name} (por {ranking_label})",
            fontsize=11,y=1.02
        )
        plt.show()
        # Tabla
        cols=["sensitivity","specificity","balanced_accuracy","gmean","tpr5"]
        for metric_name in opt_metric_names:
            if metric_name not in cols:
                cols.append(metric_name)
        rows=[]
        for name,res in results.items():
            s=res["stats"]
            row={"model":name + _star(name)}
            for c in cols:
                if c in s:
                    row[c]=f"{s[c]:.3f} ± {s.get(c+'_std',0):.3f}"
            rows.append(row)
        df_stats=pd.DataFrame(rows).set_index("model")
        print(f"\n=== Métricas CV - ordenado por {ranking_label} ===")
        if select_by!="auto" and select_by in df_stats.columns:
            ascending=False
            for spec in opt_metric_specs:
                if spec["name"]==select_by and spec["direction"]=="minimize":
                    ascending=True
                    break
            print(df_stats.sort_values(
                select_by,
                key=lambda x:x.str.split(" ± ").str[0].astype(float),
                ascending=ascending
            ).to_string())
        else:
            print(df_stats.to_string())
        print(f"\n Modelo Seleccionado: {best_name}")
        fitted={n:results[n]["pipeline"] for n in models}
        return df_stats,fitted,best_name
            





##########  COMPARACIÓN DE SAMPLERS

class SamplerComparison(CVEvaluator):
    """
    Hereda CVEvaluator añadiendo:
    -self.samplers -> catálogo
    - compare() -> samplers x modelos 
    - plot_comparison()/ print_summary()
    """
    DEFAULT_SAMPLERS= staticmethod(
        lambda strategy=0.3, seed=777:{
        "Baseline":None,
        "SMOTEENN": im.combine.SMOTEENN(random_state=seed,sampling_strategy=strategy),
        "SMOTE":im.over_sampling.SMOTE(random_state=seed,sampling_strategy=strategy),
        "ADASYN":im.over_sampling.ADASYN(random_state=seed,sampling_strategy=strategy)
    }
    )

    def __init__(self, cv, pos_label="claseSdDown",
                samplers=None, sampling_strategy=0.3, random_state=777):
        super().__init__(cv=cv,pos_label=pos_label,
                        sampling_strategy=sampling_strategy,random_state=random_state,sampler=None)
        self.samplers=(samplers if samplers is not None
                      else SamplerComparison.DEFAULT_SAMPLERS(sampling_strategy,random_state))


    #______________ comparación samplers x modelos___________________
    def compare(self,models,X,y,grids=None,select_by="auto",opt_metrics=None,
                pareto_strategy="lexicographic",pareto_weights=None):
        """
        Llama a compare_models de la superclase para cada sampler.
        Devuelve un dataframe MultiIndex (sampler,model)
        """
        all_rows=[]
        for sampler_name,sampler in tqdm(self.samplers.items(),desc="Samplers"):
            df_models,_,_=self.compare_models(models,X,y,grids=grids,
                                              sampler=sampler,select_by=select_by,
                                              opt_metrics=opt_metrics,
                                              pareto_strategy=pareto_strategy,
                                              pareto_weights=pareto_weights)
            for model_name,row in df_models.iterrows():
                d= row.to_dict()
                d["sampler"]=sampler_name
                d["model"]=model_name.replace(chr(171),"")
                all_rows.append(d)
        return pd.DataFrame(all_rows).set_index(["sampler","model"])

    #___________________PLOTS__________________-

    @staticmethod
    def plot_comparison(df,
                       metrics_cols=("sensitivity","specificity","balanced_accuracy","gmean"),
                       figsize_per_metric=(4,4)):
        samplers=df.index.get_level_values("sampler").unique().tolist()
        mods=df.index.get_level_values("model").unique().tolist()
        n=len(metrics_cols)
        w,h=figsize_per_metric
        colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        #(groups,labels) define qué va en eje X y qué color
        for groups,labels,title in [(mods,samplers,"Efecto del sampler por modelo"),
                                    (samplers, mods,"Efecto del modelo por sampler")]:
            fig,axes=plt.subplots(1,n,figsize=(w*n,h))
            fig.suptitle(title,fontsize=13,y=1.01)
            if n==1:
                axes=[axes]
            x= np.arange(len(labels))
            width= 0.8/ len(groups)
            for ax,met in zip(axes,metrics_cols):
                for i,grp in enumerate(groups):
                    get= lambda s,m: float(df.loc[(s,m),met].split("±")[0])
                    get_std=lambda s,m: float(df.loc[(s,m),met].split("±")[1])
                    if groups is mods:
                        
                        vals=[get(s,grp) for s in labels]
                        errs=[get_std(s,grp) for s in labels]
                    else:
                        vals=[get(grp,m) for m in labels]
                        errs=[get_std(grp,m) for m in labels]
                    offset=(i-len(groups)/2+0.5)*width
                    ax.bar(x+offset,vals,width,yerr=errs,capsize=3,
                          label=grp,color=colors[i%len(colors)],alpha=0.85)
                ax.set_title(met.replace("_"," ").title())
                ax.set_xticks(x)
                ax.set_xticklabels(labels,rotation=30,ha="right",fontsize=8)
                ax.set_ylim(0,1.05)
                ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1))
                ax.grid(axis="y",lw=0.4,alpha=0.5)
                ax.legend(fontsize=7,ncol=2)
            plt.tight_layout()
            plt.show()
