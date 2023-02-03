function strOut = polyquant_grad(specData,A,At,I0,rho,y,ind,scatFun,subSet,w)
% This function calculates the gradient, objective function unless using
% OS, and the scatter if calculated on the fly.
projSet = cell(length(specData.hinge)-1,2);
mask = cell(length(specData.hinge)-1,1);
projSet{1,2} = 0;
for k = 1:length(specData.hinge)-1
    mask{k} = double(rho > specData.hinge(k) & rho < specData.hinge(k+1));
    projSet{k,1} = A(mask{k}.*rho,ind);
    if k>1
        projSet{k,2} = A(mask{k},ind);
    end
end
specProb = specData.spectrum./sum(specData.spectrum(:));
    
mainFac = zeros(size(y));
hingeFac = cell(length(specData.hinge)-1);
for k = 1:length(specData.hinge)-1
    hingeFac{k} = zeros(size(y));
end

if length(specData.hinge)>2  % to bodge error for one linear fit
    s = scatFun(I0,projSet{1,1},projSet{2,1},projSet{2,2},rho,subSet,specData.knee);
else
    s = scatFun(I0,projSet{1,1},projSet{1,1},projSet{1,2},rho,subSet,specData.knee);
end
for k = 1:length(specData.spectrum)
    linSum = zeros(size(y));
    for l = 1:length(specData.hinge)-1
        linSum = linSum+specData.knee(1,l,k)*projSet{l,1}...
                       +specData.knee(2,l,k)*projSet{l,2};
    end
    tmp = specProb(k).*exp(-linSum);
    mainFac = mainFac+tmp;
    for l = 1:length(specData.hinge)-1
        hingeFac{l} = hingeFac{l}+tmp*specData.knee(1,l,k);
    end
end
mainFac = I0.*mainFac;

deriFac = w(y./(mainFac+s)-1);

out = zeros(size(rho));
for l = 1:length(specData.hinge)-1
    out = out+mask{l}.*At(I0.*hingeFac{l}.*deriFac,ind);
end

strOut.grad = out;
strOut.objFac = mainFac;
strOut.s = s;
end