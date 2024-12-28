
module main (
    input AHL,
    input Benzoate,
    output GFP
);
    assign GFP = !(pTac && pTet);  // Updated with promoters from JSON
endmodule
