clear all






for i=1:3
    im = double(imread(sprintf('input%d.png',i)));
    ime = double(imread(sprintf('bss%d.png',i)));
    for j=1:3.
        imtmp = im(:,:,j);
        ims = imtmp(:);
        srct(:,j) = ims;
        imetmp = ime(:,:,j);
        imes = imetmp(:);
        estt(:,j) = imes;
    end
    src(i,:,:) = srct;
    est(i,:,:) = estt;
end

[sdr isr sir sar perm] = bss_eval_images(est,src);
fprintf('SDR:%g\nISR:%g\nSIR:%g\nSAR:%g\n',mean(sdr),mean(isr),mean(sir),mean(sar));
