function resp = vessel_response_demo(green, scales)
%VESSEL_RESPONSE_DEMO Multi-scale white top-hat on inverted green
%   (demonstration mirror of the Phase 4A baseline idea). Output 0..1
%   RESPONSE map, never called a probability.
    if nargin < 2, scales = [3 5 9 15]; end
    inv = 1 - im2single(green);
    resp = zeros(size(inv), 'single');
    for s = scales
        se = strel('disk', s);
        resp = max(resp, imtophat(inv, se));
    end
    m = max(resp(:));
    if m > 0, resp = resp / m; end
end
