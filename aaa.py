import matlab2python as mpars
# --- Convert a matlab file
mlines="""#function out = offset_weight(proj,cg)
    % Offset weighting for half-fan case from [G. Wang, Med Phys. 2002]
    out = proj;
    us = ((cg.ns/2-0.5):-1:(-cg.ns/2+0.5))*cg.ds - cg.offset_s*cg.ds;
    overlap = max(us);
    overLoc = sum(abs(us)<=overlap);
    replaceLoc = 1:overLoc;
    denom = 2*atan(overlap/cg.dsd);
    num = pi*atan(us(replaceLoc)/cg.dsd);
    %weightArray = 1-cos(linspace(0,pi/2,overLoc)).^2;
    weightArray = 1-0.5*(sin(num./denom)+1);
    weightMat = repmat(weightArray',1,cg.nt);
    replaceLoc = 1:size(weightMat,1);

    for k = 1:size(proj,3)
        out(end-replaceLoc+1,:,k) = proj(end-replaceLoc+1,:,k).*weightMat;
    end
end
"""

pylines = mpars.matlablines2python(mlines, output='stdout')
