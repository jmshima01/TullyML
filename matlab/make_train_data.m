function train_data = make_train_data(fname)

fpath = '';
slash_ind = findstr(fname, '\');
if(~isempty(slash_ind))
    fpath = fname(1:slash_ind(end));
    fname = fname(slash_ind(end)+1:end);
end

b=load("bench_acc_data.mat");
s=load("squat_acc_data.mat");
c=load("curl_acc_data.mat");
d=load("deadlift_acc_data.mat");
o=load("overhead_acc_data.mat");

b_range=[1200:3450];
s_range=[750:2850];
c_range=[1:4100];
d_range=[1420:4500];
o_range=[1280:4200];

b_xacc = b.xacc(b_range); 
b_yacc = b.yacc(b_range);
b_zacc = b.zacc(b_range);

s_xacc = s.xacc(s_range);
s_yacc = s.yacc(s_range);
s_zacc = s.zacc(s_range);

c_xacc = c.xacc(c_range);
c_yacc = c.yacc(c_range);
c_zacc = c.zacc(c_range);

d_xacc = d.xacc(d_range);
d_yacc = d.yacc(d_range);
d_zacc = d.zacc(d_range);

o_xacc = o.xacc(o_range);
o_yacc = o.yacc(o_range);
o_zacc = o.zacc(o_range);

exerciseTypes = categorical(["bench", "overhead", "squat", "deadlift", "curl"]);
percentTrainingSamples = .80;
percentValidationSamples = .10;
percentTestSamples = .10;

%cut up into snippets
blksize = 80;
ov = 0.8;
[B, Nb] = block_data(b_xacc, b_yacc, b_zacc, blksize, ov);
[D, Nd] = block_data(d_xacc,d_yacc,d_zacc, blksize, ov);
[S, Ns] = block_data(s_xacc, s_yacc, s_zacc, blksize, ov);
[O, No] = block_data(o_xacc, o_yacc, o_zacc, blksize, ov);
[C, Nc] = block_data(c_xacc, c_yacc, c_zacc, blksize, ov);

train_data = zeros(3,blksize,1,Nb+Ns+Nd+Nc+No);
train_labels = categorical(exerciseTypes(1));
train_labels = repmat(train_labels,Nb+Ns+Nd+Nc+No,1);
ind = [1:Nb];
train_data(:,:,:,ind) = B;
train_labels(ind) = exerciseTypes(1);
ind = [1:Nd] + Nb;

train_data(:,:,:,ind) = D;
train_labels(ind) = exerciseTypes(4);
ind = [1:Ns] + Nb + Nd;

train_data(:,:,:,ind) = S;
train_labels(ind) = exerciseTypes(3);
ind = [1:Nc] + Nb + Nd + Ns;

train_data(:,:,:,ind) = C; 
train_labels(ind) = exerciseTypes(5);
ind = [1:No] + Nb + Nd + Nc + Ns;

train_data(:,:,:,ind) = O;
train_labels(ind) = exerciseTypes(2);

Nt = length(train_labels);

ii = randperm(Nt);

N1 = round(Nt*percentTrainingSamples);
N2 = round(Nt*percentValidationSamples);
N3 = round(Nt*percentTestSamples);

ind = ii(1:N1);

dataTraining = train_data(:,:,:,ind);
dataTrainingLabel = train_labels(ind);

ind = ii([1:N2]+N1);
dataValidation = train_data(:,:,:,ind);
dataValidationLabel = train_labels(ind);

ind = ii([1:N3]+N1+N2);
dataTest = train_data(:,:,:,ind);
dataTestLabel = train_labels(ind);

save([fpath,fname,'_train_data.mat'], 'dataTraining', 'dataTrainingLabel', 'exerciseTypes', '-v7.3');
save([fpath,fname, '_val_data.mat'],   'dataValidation', 'dataValidationLabel', '-v7.3');
save([fpath,fname, '_test_data.mat'],  'dataTest', 'dataTestLabel', 'exerciseTypes', '-v7.3');

disp(['Saved all data to ',fpath, fname]);

%-----------------------------------------------------
function [D, Nblks] = block_data(x,y,z, blksize, ov)

    Nov = round(blksize*ov);
    Nshift = blksize - Nov;
    Nblks = floor(1 + (length(x) - blksize)/Nshift);
    
    ind = [1:blksize];
    D = zeros(3,blksize,Nblks);
    for i=1:Nblks
        D(1,:,i) = x(ind);
        D(2,:,i) = y(ind);
        D(3,:,i) = z(ind);
        ind=ind+Nshift;
    end





