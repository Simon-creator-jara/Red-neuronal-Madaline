function [Yk,ecm,C,W] = Seno(alfa,entradas,deseados,W,C,entrena)
nd=size(deseados,2);
ns=size(deseados,1);
no=size(W,1);

Hj=zeros(no+1,1);
%%Creamos salida y ecm
Yk=zeros(ns,nd);
ecm=zeros(ns,1);

for i =1:nd
    %%Calculamos la agregación AJ de capa oculta
    Aj=W*entradas(:,i);
    
    %%Guardamos BIAS de capa oculta
    Hj(1)=1;
    
    %%Calculamos fn de activación capa oculta
    Hj(2:end)=sin(Aj); %Fn seno
    
    %%Calculamos agregación de capa de salida 
    Ak=C*Hj;
    
    %%Calculamos Fn de activación de capa de salida 
    Yk(:,i)=1./(1+exp(-Ak));%%Fn sigmoidal
    
    %%Calculamos el error 
    Ek=deseados(:,i)-Yk(:,i);
    
    %%Calculamos el ECM
    ecm(:)=ecm(:)+(Ek.^2)./2;
    
    
    %%Entrenamos 
    if entrena==1
        %%Calculamos sensitividad de capa de salida 
        ds= Ek.*(Yk(:,i).*(1-Yk(:,i)));
        
        %%Calculamos sensitividad de la capa oculta
        dh =(ds'*C(:,2:end)).*cos(Aj)';
        
        %%Actualizamos pesos C 
        C=C+alfa*(ds*Hj');
        
        %%Actualizamos Wij
        W = W + alfa*((dh'*entradas(:,i)'));
        
        
    end
    
end


end