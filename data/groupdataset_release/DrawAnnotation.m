function DrawAnnotation(imfile, annofile, outfile)

type_color = {'b','g','r'};

img = imread(imfile);

load(annofile);

% visualize humans
[row,col,~] = size(img);
figure('Position',[0 0 col row]);
imagesc(img); axis image; axis off; hold on
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf,'color',[1 1 1]);
set(gca,'color',[1 1 1]);
if size(img,3) == 1
    colormap(gray);
end

for id = 1:length(hmns)
    
    for i = 1:length(hmns{id})
        
        entr_poly = hmns{id}(i).entr_poly;
        plot([entr_poly(:,1); entr_poly(1,1)],[entr_poly(:,2); entr_poly(1,2)], 'linewidth', 4, 'Color', 'w');
        rectangle('position', hmns{id}(i).entr_bbs, 'edgecolor', type_color{id}, 'linewidth', 2);
        
        pt1 = [hmns{id}(i).entr_bbs(1) + hmns{id}(i).entr_bbs(3) / 2, ...
            hmns{id}(i).entr_bbs(2) + hmns{id}(i).entr_bbs(4) / 2];
        
        len = hmns{id}(i).entr_bbs(4) * 0.2;
        
        drc = [cos(hmn_poses{id}(i).az+pi/2) sin(hmn_poses{id}(i).az+pi/2)];
        origlen = norm(drc);
        pt2 = pt1 + drc*(len/origlen);
        scatter(pt1(1), pt1(2), 'co')
        plot([pt1(1) pt2(1)], [pt1(2) pt2(2)], 'linewidth', 2, 'color', 'c');

    end
    
end

hold off;

if nargin >= 3
    print('-djpeg', [outfile '_hmn'], '-r0');
end

% visualize groups
[row,col,~] = size(img);
figure('Position',[0 0 col row]);
imagesc(img); axis image; axis off; hold on
set(gca, 'Position', [0 0 1 1]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf,'color',[1 1 1]);
set(gca,'color',[1 1 1]);
if size(img,3) == 1
    colormap(gray);
end

cnt = 1;
for iter_type = 1:length(hmns)
    for iter_id = 1:length(hmns{iter_type})
        cnt = cnt+1;
    end
end
plot_count = zeros(cnt-1,1);

for id = 1:length(grps)
    
    for i = 1:length(grps{id})
        
        member = grps{id}(i).member;
        
        for m = 1:size(member,1)
            bbs = hmns{member(m,2)}(member(m,3)).entr_bbs;
            rectangle('position', bbs, 'edgecolor', type_color{member(m,2)}, 'linewidth', 2);
            text(bbs(1)+bbs(3)/2-10, bbs(2)+15+22*plot_count(member(m,1)), ...
                [' ' num2str(id) '-' num2str(i) ' '], ...
                'Color','w', ...
                'backgroundcolor',type_color{member(m,2)});
            plot_count(member(m,1)) = plot_count(member(m,1))+1;
        end

    end
    
end

% visualize outliers
for i = 1:size(outliers,1)
    bbs = hmns{outliers(i,2)}(outliers(i,3)).entr_bbs;
    rectangle('position', bbs, 'edgecolor', 'y', 'linewidth', 2);
end
hold off;

if nargin >= 3
    print('-djpeg', [outfile '_grp'], '-r0');
end

end
