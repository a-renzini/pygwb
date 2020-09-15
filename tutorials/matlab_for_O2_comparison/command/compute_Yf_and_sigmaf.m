function compute_stats(dir,fileprefixes,outputFileName,h0,bias,notches)
% Computes point estimate and error bar given pointers to the sensitivity integrand and point esitmate integrand and a notchlist
% Written by Andrew Matas (andrew.matas@ligo.org) Mar 6, 2017

if nargin==5  
  notches=[];
end


% frequency band
tmp=load([dir fileprefixes{1} '_ptEstIntegrand.dat']);
freq=tmp(:,2);

df=freq(2)-freq(1);
fmax=max(freq);
fmin=min(freq);
nbins=(fmax-fmin)/df+1;

% matrices to store the narrowband stats for each detector pair
ptEst_I=zeros([length(fileprefixes) nbins]);
sigma_I=zeros([length(fileprefixes) nbins]);


% construct narrowband stats for each detector pair
for ii=1:length(fileprefixes)

   % read in data
   tmp=load([dir fileprefixes{ii} '_ptEstIntegrand.dat']);
   freq=tmp(:,2);
   ptEstInt_f=tmp(:,3); %just the real part
   ptEstIntIm_f=tmp(:,4);

   tmp=load([dir fileprefixes{ii} '_sensIntegrand.dat']);
   sensInt_f=tmp(:,3);

   df=freq(2)-freq(1);


   % compute narrowband statistics for this pair
   sigma_f=1./sqrt(sensInt_f*df);
   sigma_tot=sqrt(1/sum(sigma_f.^(-2)));
   ptEst_f=2/sigma_tot^2 * (ptEstInt_f+i*ptEstIntIm_f)./sensInt_f;

    % remove the notched frequencies (replace Y with 0 and sigma with Inf in the notched bins)
    cut_f=isnan(ptEst_f);

    ptEst_f(cut_f)=0;
    sigma_f(cut_f)=Inf;

   % add the narrowband stats for this detector pair to the matrix
   ptEst_I(ii,:)=ptEst_f';
   sigma_I(ii,:)=sigma_f';

end

% construct the full (combined) narrowband stats by combining detector pairs
for i_ff=1:nbins
   ptEst_ff(i_ff)=sum(ptEst_I(:,i_ff).*sigma_I(:,i_ff).^(-2))./sum(sigma_I(:,i_ff).^(-2));
   sigma_ff(i_ff)=1./sqrt(sum(sigma_I(:,i_ff).^(-2),1));
end

% remove any nans (comes from places where sigma=Inf in the above lines)
ptEst_ff(isnan(ptEst_ff))=0;

% apply notches
for i_ff=1:nbins
   for j_pp=1:length(notches)
     if freq(i_ff)==notches(j_pp)
	      ptEst_ff(i_ff)=0;
              sigma_ff(i_ff)=Inf;
     end
   end
end
		         

% apply Hubble and bias
sigma_ff=sigma_ff * bias/h0^2;
ptEst_ff = ptEst_ff / h0^2;

% compute broadband stats
Ynum=sum(ptEst_ff./(sigma_ff.*sigma_ff));
Yden=sum(1./(sigma_ff.*sigma_ff));
Y=Ynum/Yden;
sigma=1/sqrt(sum(1./(sigma_ff.*sigma_ff)));


% print results to screen 
fprintf('Final broadband stats using h0=%e, bias=%e\n',h0,bias)
fprintf('Y=%e +/- %e\n',Y,sigma)
fprintf('SNR=%e\n',Y/sigma)
fprintf('\n')


% save results as mat file
save(outputFileName,'freq','ptEst_ff','sigma_ff')

% save results as text file
fid=fopen([outputFileName '.dat'],'w');
fprintf(fid,'%%Frequency\tReal Y(f)\tImag Y(f)\tSigma(f)\n');
for ii=1:length(freq)
    fprintf(fid,'%e\t%e\t%e\t%e\n',[freq(ii) real(ptEst_ff(ii)) imag(ptEst_ff(ii)) sigma_ff(ii)]');
end

