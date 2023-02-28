function feat = computefeature(structdis)

% Input  - MSCn coefficients
% Output - Compute the 18 dimensional feature vector 

feat          = [];

shifts                   = [1 0];

for itr_shift =1:1
 
shifted_structdis        = circshift(structdis,shifts(itr_shift,:));
pair                     = structdis(:).*shifted_structdis(:);
[alpha, betal, betar]      = estimateaggdparam(pair);
meanparam                = (betar-betal)*(gamma(2/alpha)/gamma(1/alpha));                       
feat                     = [feat;alpha;meanparam;betal;betar];

end


