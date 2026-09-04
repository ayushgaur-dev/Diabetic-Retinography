function rgb = load_fundus(path, targetWidth)
%LOAD_FUNDUS Read a fundus photograph as uint8 RGB, optional resize.
%   rgb = load_fundus(path) keeps native resolution.
%   rgb = load_fundus(path, W) rescales width to W (aspect preserved).
    assert(isfile(path), 'Image not found: %s', path);
    img = imread(path);
    if ndims(img) == 2
        rgb = repmat(img, [1 1 3]);
    elseif size(img, 3) == 4
        rgb = img(:, :, 1:3);   % drop alpha
    else
        assert(size(img, 3) == 3, 'Expected RGB image: %s', path);
        rgb = img;
    end
    if nargin >= 2 && ~isempty(targetWidth) && size(rgb, 2) ~= targetWidth
        s = targetWidth / size(rgb, 2);
        rgb = imresize(rgb, [round(size(rgb, 1) * s), targetWidth]);
    end
end
