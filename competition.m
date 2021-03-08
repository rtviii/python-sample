%Rewritten October 20th 2020
iterations = 100000000;
%%%%%%%%%%%%Starting Data%%%%%%%%%%%%%%%%
%%%Data about population
inds = 1000; %Our population will have space for N*2 individuals
genes = 3;
traits = 3;
%%%Space for individuals 
contributions = NaN(traits,genes,inds*2);
alleles = NaN(inds*2,genes);
%%%Starting everyone the same:
beginalleles = [1 1 1];
contritype1 = [0.33 0.33 0.33; 0.33 0.33 0.33 ; 0.33 0.33 0.33];
contritype2 = [1 0 0; 0 1 0; 0 0 1];
angularcoef = [1 1 1];

%%Starting individuals
for j = 1:(inds)
    if rem(j,2)
      contributions(:,:,j) = contritype2;
    else
       contributions(:,:,j) = contritype1;
    end
    alleles(j,:) = beginalleles;
end

%for j = 1:(inds/3)
 %   contributions(:,:,j) = contritype1;
  %   alleles(j,:) = beginalleles;
%end
%for k = (inds/3):2*(inds/3)
 %   contributions(:,:,k) = contritype2;
  %   alleles(k,:) = beginalleles;
%end
%for yi = 2*(inds/3):inds
 %   contributions(:,:,yi) = contritype3;
  %   alleles(yi,:) = beginalleles;
%end    

%%Starting data
phenotypes = phecalculator(contributions,alleles);
fitness = fitcalculator(phenotypes, angularcoef);
dyingvector = penalti(contributions);
probabilities = ALLprobcalculator(contributions,alleles,angularcoef);

nasce = 0;
morre = 0;
%fid = fopen( 'results.txt', 'wt' );

%pheno = zeros(2,iterations);
%%Simulation


for i = 1:iterations
    N = length(probabilities); 
    dB = choosing(probabilities);
    if isnan(probabilities(dB))        
            continue          
    end
    if dB <= N/2     
            nasce = nasce +1;  
            [contributions,alleles,probabilities,fitness,dyingvector] = birth(dyingvector,contributions,alleles,fitness,dB, angularcoef);  
    else     
            morre = morre +1;         
            [contributions,alleles,probabilities,fitness,dyingvector] = death(contributions,alleles,probabilities,dB,fitness,dyingvector);      
    end
    Fi(i) = nanmean(fitness); 
    Pol(i) = nanmean(sum(vecnorm(contributions,2,2))); 
    %writematrix(M, "results.txt");
    %pheno(:,i) = nanmean(vecnorm(contributions,2,2),3);  
end


%%Functions

%%1. Calculating data for All Individuals
%Phenotypes
function phenotypes = phecalculator(contributions, alleles)
    %alleles are transposed
    transpose = [0,1];
    phenotypes = tmult(contributions, alleles, transpose);
    phenotypes = phenotypes(:,1,:);
    [ind, genes] = size(alleles);
    phenotypes = reshape(phenotypes,[genes,ind]);
    %~isnan(phenotypes)
    %phenotypes = phenotypes(:,1,:);
    %phenotypes = squeeze(phenotypes) %if more than 1 gene 1 trait, individuals are columns
end
  
%Fitness
function fitness = fitcalculator(phenotypes, angularcoef)  
   fitness = angularcoef*(phenotypes.^2) + 10;  
end

%Cost - prob of dying
function dyingvector = penalti(contributions)
    %how bad is it?
    %[t,~,~] = size(contributions);
    
    strength = 0;
    N = sum(~isnan(contributions(1,1,:)));
    
    pen = 0.01 * (1 + strength*(sum(vecnorm(contributions,2,2))));
    %pen = 0.01 * (1 + strength*(t - sum(vecnorm(contributions,2,2))));
    dyingvector = reshape(pen,[],size(pen,2),1)';
    dyingvector = dyingvector*N;
end

%Creating All the probabilities 
function [probabilities,fitness,dyingvector] = ALLprobcalculator(contributions,alleles, angularcoef)
    [nind,~] = size(alleles);
    probabilities = NaN(nind,2);
    phenotypes = phecalculator(contributions,alleles);
    %2nd collumn is death
    dyingvector = penalti(contributions);
    %1st collumn is reproduction
    fitness = fitcalculator(phenotypes, angularcoef);
    ototal = nansum(fitness) + nansum(dyingvector);
    dyingvector = log(dyingvector+1)/ototal;
    probbirth = log(fitness+1)/ototal;
    probabilities(:,2) = dyingvector;
    probabilities(:,1) = probbirth;
    probabilities = probabilities(:)';
    %probabilities(~isnan(probabilities));
    %average = mean(fitness);
    %Scale probabilities so that its range is in the interval [0,1].
end

%Tensor multiplication
function A = tmult(A, B, transpose)
    szB = [size(B) 1];
    szA = [size(A) 1];
    if nargin < 3
        transpose = 0;
    else
        transpose = [transpose(:); 0];
        transpose = [1 2] * (transpose(1:2) ~= 0);
        if transpose == 3
            % Permutation required. Choose the matrix which permutes fastest.
            pa = any(szA(1:2) == 1);
            pb = any(szB(1:2) == 1);
            if pa || (~pb && numel(A) < numel(B))
                if ~pa
                    p = 1:numel(szA);
                    p(1:2) = p([2 1]);
                    A = permute(A, p);
                end
                szA(1:2) = szA([2 1]);
                transpose = 2;
            else
                if ~pb
                    p = 1:numel(szB);
                    p(1:2) = p([2 1]);
                    B = permute(B, p);
                end
                szB(1:2) = szB([2 1]); 
                transpose = 1;    
            end  
        end  
    end  
    switch transpose  
        case 0
            % No transposes
            A = reshape(A, szA([1:2 end 3:end-1]));
            B = reshape(B, szB([end 1:end-1]));
            dim = 2;
            szB(1) = szA(1);
        case 1
            % First matrix transposed       
            A = reshape(A, szA([1:2 end 3:end-1]));    
            B = reshape(B, szB([1 end 2:end])); 
            dim = 1;
            szB(1) = szA(2);   
        case 2 
            % Second matrix transposed    
            A = reshape(A, szA([1 end 2:end]));
            B = reshape(B, szB([end 1:end-1]));
            dim = 3;          
            szB(2) = szB(1);          
            szB(1) = szA(1);          
    end
    % Compute the output            
    A = sum(bsxfun(@times, A, B), dim);        
    % Reshape to expected size    
    clszA = [szA ones(1, numel(szB)-numel(szA))];           
    szB = [szB ones(1, numel(szA)-numel(szB))];    
    szB(3:end) = max(szB(3:end), szA(3:end));
    A = reshape(A, szB); 
end


%2. Single individual functions
%Fitness
function onefitness = onefit(singlephenotype, angularcoef) 
    onefitness = angularcoef*(singlephenotype.^2) + 10;
end


%%3.Event-type functions

%Choosing an Event
function chosen = choosing(probabilities)
    choices = probabilities(~isnan(probabilities)); 
    n = length(choices); 
    if any(choices)      
        dB = randsample(n,1,true,choices);     
    else  
        dB = randsample(n,1,true);     
    end
    Truenan=(isnan(probabilities)); 
    sumN=cumsum(Truenan);
    numberofnan=sumN(~isnan(probabilities));
    chosen = numberofnan(dB) + dB;
    %la= length(probabilities);
    %nanmean(probabilities(1:la/2)) >= probabilities(chosen)
end
%Death Event
 function [contributions,alleles,probabilities,fitness,dyingvector] = death(contributions,alleles,probabilities,dB,fitness,dyingvector)
    [side1,side2,~] = size(contributions);
    [~,N] = size(probabilities);
    probabilities(dB) = NaN;
    dB = dB - N/2;
    contributions(:,:,dB) = NaN(side1,side2);
    alleles(dB,:) = NaN(side2,1);
    probabilities(dB) = NaN;
    fitness(dB) = NaN;
    dyingvector(dB) = NaN;
    %recalculaitng death prob
    dyingvector = penalti(contributions);
    ototal = nansum(fitness) + nansum(dyingvector);
    dyingvector = log(dyingvector +1)/ototal;
    probbirth = log(fitness +1)/ototal;
    probs(:,2) = dyingvector;
    probs(:,1) = probbirth;
    probabilities = probs(:)';
 end
%Birth Event
function [contributions,alleles,probabilities,fitness,dyingvector] = birth(dyingvector, contributions,alleles, fitness,dB, angularcoef)
    cont = contributions(:,:,dB);
    alle = alleles(dB,:);
    emptyspace = find(isnan(squeeze(contributions(1,1,:))));
    emptyspace = emptyspace(1);
    contributions(:,:,emptyspace) = cont;
    alleles(emptyspace,:) = alle;
    newpheno = cont*alle';
    fitness(emptyspace) = onefit(newpheno, angularcoef);
    dyingvector = penalti(contributions);
    ototal = nansum(fitness) + nansum(dyingvector);
    dyingvector = log(dyingvector+1)/ototal;
    probbirth = log(fitness+1)/ototal;
    probs(:,2) = dyingvector;
    probs(:,1) = probbirth;
    probabilities = probs(:)';
end 
