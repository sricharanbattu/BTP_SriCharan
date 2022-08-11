%%tic
pkg load statistics
%%%%%%pkg rebuild -noauto oct2mat
%%%%%%%%%LMS ALGORITHM FOR BILINEAR SYSTEMS  %%%%%%%%%%%%%%%%%%%
N=30000;
L=64;
M=8;
sigmax=0.1;
sigmaw=0.0001;
sigmah=2.5;
sigmag=2.5;
h_true = normrnd(0,sigmah,L,1);
g_true = normrnd(0,sigmag,M,1);
f_true = kron(g_true,h_true);

norm_h = norm(h_true);
norm_g = norm(g_true);
norm_f = norm(f_true);

h_transpose = transpose(h_true);
g_transpose = transpose(g_true);

%%%Implement LMS - BF Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%
IL = eye(L);
IM = eye(M);
mu_h = 0.001;
mu_g = 0.001;
h_cap = zeros(L,1);
h_cap(1) = 1;
g_cap=ones(M,1);
g_cap= g_cap/M;

NPLM_hvec=zeros(N,1);
NPLM_gvec=zeros(N,1);

alpha_h=0.6;
alpha_g=0.6;
del_hcap=sigmax^2;
del_gcap=sigmax^2;
Nh_cap=h_cap;
Ng_cap=g_cap;
NNPLM_hvec=zeros(N,1);
NNPLM_gvec=zeros(N,1);


for i=1:N
  X_n= normrnd(0,sigmax,L,M);
  w  = normrnd(0,sigmaw);
  d_n = transpose(h_true)*X_n*g_true + w;
  x_vec = reshape(X_n,[],1);
  %%%%%%%%%%%%%%%%%%%%%%%%LMS IMPLEMENTATION%%%%%%%%%%%%%%%%%%%%%%
  x_gcap= transpose(kron(g_cap,IL))*x_vec;
  x_hcap= transpose(kron(IM,h_cap))*x_vec;
  
  e_gcap = d_n - transpose(h_cap)*x_gcap;
  e_hcap = d_n - transpose(g_cap)*x_hcap;
  h_cap  = h_cap + mu_h*e_gcap*x_gcap;
  g_cap  = g_cap + mu_g*e_hcap*x_hcap;
                %%%%%%%%%%CALCULATING NPM'S %%%%%%%%%%%%%%%%%%%%%%%
  norm_hcap = norm(h_cap);
  norm_gcap = norm(g_cap);
  NPLM_h    = 1- ((h_transpose*h_cap)/(norm_h*norm_hcap))^2;
  NPLM_g    = 1- ((g_transpose*g_cap)/(norm_g*norm_gcap))^2;
  NPLM_hvec(i)=10*log10(NPLM_h);
  NPLM_gvec(i)=10*log10(NPLM_g);
  %%%%%%%%%%%%%%%%%%%%%NLMS IMPLEMENTATION%%%%%%%%%%%%%%%%%%%%%
  Nx_gcap= transpose(kron(Ng_cap,IL))*x_vec;
  Nx_hcap= transpose(kron(IM,Nh_cap))*x_vec;
  Ne_gcap = d_n - transpose(Nh_cap)*Nx_gcap;
  Ne_hcap = d_n - transpose(Ng_cap)*Nx_hcap;
  Nh_cap  = Nh_cap + alpha_h*Ne_gcap*Nx_gcap/(transpose(Nx_gcap)*Nx_gcap  + del_hcap);
  Ng_cap  = Ng_cap + alpha_g*Ne_hcap*Nx_hcap/(transpose(Nx_hcap)*Nx_hcap  + del_gcap);
                %%%%%%%%%%CALCULATING NPM'S %%%%%%%%%%%%%%%%%%%%%%%
  Nnorm_hcap = norm(Nh_cap);
  Nnorm_gcap = norm(Ng_cap);
  NNPLM_h    = 1- ((h_transpose*Nh_cap)/(norm_h*Nnorm_hcap))^2;
  NNPLM_g    = 1- ((g_transpose*Ng_cap)/(norm_g*Nnorm_gcap))^2;
  NNPLM_hvec(i)=10*log10(NNPLM_h);
  NNPLM_gvec(i)=10*log10(NNPLM_g);
  
  
endfor

%estimating eta
eta=0;
for i=1:M
  if(g_true(i)!=0)
    eta=eta + (g_cap(i)/g_true(i));
  else
    eta = eta +(g_cap(i)/0.00001);
  endif
  
  
 endfor
 
 eta = eta/M;
 %%%%estimating 1/eta
 etainv=0;
 for i=1:L
   if(h_true(i)!=0)
    etainv=etainv + (h_cap(i)/h_true(i));
  else
    etainv = etainv +(h_cap(i)/0.00001);
  endif
   
  endfor
  
  etainv = etainv/L;
  normal_normh= etainv*norm_h;
  normal_normg=eta*norm_g;
  delta = 2 - (sigmax^2)*(mu_h*L*normal_normg^2  +  mu_g*M*normal_normh^2);
  mh_inf= (mu_h*L* sigmaw^2)/(delta*normal_normh^2);
  mg_inf= (mu_g*M* sigmaw^2)/(delta*normal_normg^2);
  mh_inf=10*log10(mh_inf);
  mg_inf=10*log10(mg_inf);
  
  %%%%%%%fileoperations for LMS
f=fopen('LMS.txt','w');
fprintf(f,"%f,%f",mh_inf,mg_inf);
fdisp(f,',');
for i=1:N
  fprintf(f,"%f,%f",NPLM_hvec(i),NPLM_gvec(i));
  fdisp(f,',');
endfor
fclose(f);

  %%%%%%%%%%%fleoperations for NLMS
Ndelta = sigmax^2*norm_f^2*(2-alpha_g-alpha_h);
Nmg_inf=(alpha_g* sigmaw^2)/Ndelta;
Nmh_inf=(alpha_h* sigmaw^2)/Ndelta;
Nmh_inf=10*log10(Nmh_inf);
Nmg_inf=10*log10(Nmg_inf);
Nf=fopen('NLMS.txt','w');
fprintf(Nf,"%f,%f",Nmh_inf,Nmg_inf);
fdisp(Nf,',');
for i=1:N
  fprintf(Nf,"%f,%f",NNPLM_hvec(i),NNPLM_gvec(i));
  fdisp(Nf,',');
endfor
fclose(Nf);
%%tok