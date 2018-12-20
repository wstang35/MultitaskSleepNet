function  save_result_txt(yt, yh, onset, folder, fname)
% 保存检测结果为EDFbrowser可读的ASCII Annotation
filename = [folder, fname, '.txt'];
fid=fopen(filename,'w+');
title='Onset,Duration,Annotation';
fprintf(fid,'%s\r\n',title);
num_to_stage = ['Stage ?';'Stage W';'Stage R';'Stage 1';'Stage 2';'Stage 3'];
for i=1:size(yt)
    fprintf(fid,'%s','+');
    fprintf(fid,'%g',onset + 30*(i-1));
    fprintf(fid,'%s',',');
    fprintf(fid,'%g',30);
    fprintf(fid,'%s',',');
    if yt(i) == yh(i)
        result = 'Correct';
    else
        result = 'Error';
    end
    fprintf(fid,'%s\r\n',[result, ' True: ',num_to_stage(yt(i)+1, :),' Pred: ',num_to_stage(yh(i)+1, :)]);
end
fclose(fid);