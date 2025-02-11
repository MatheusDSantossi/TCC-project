import React, { useEffect, useState } from "react";

import "../App.css";

// Component

const SynonymDetails = ({ setProductSynonyms, resetKey }) => {
  const [productLocalSynonyms, setProductsLocalSynonyms] = useState([]);
  const [inputValue, setInputValue] = useState("");

  useEffect(() => {
    setProductsLocalSynonyms([]);
    setInputValue("");
  }, [resetKey]);

  useEffect(() => {
    setProductSynonyms(productLocalSynonyms);
  }, [productLocalSynonyms, setProductSynonyms]);

  const handleKeyDown = (e) => {
    if (e.key === " " && inputValue.trim()) {
      e.preventDefault();
      if (!productLocalSynonyms.includes(inputValue.trim())) {
        setProductsLocalSynonyms((prev) => [...prev, inputValue.trim()]);
      }
      setInputValue("");
    }
    if (e.key === "Backspace" && !inputValue) {
      setProductsLocalSynonyms((prev) => prev.slice(0, -1));
    }
  };
  // return isLoading ? (
  return (
    <div className="mb-5">
      <label
        htmlFor="productSynonyms"
        className="block text-sm font-semibold text-white"
      >
        Sinônimos para o Produto
      </label>
      <div
        className="flex flex-wrap items-center w-full p-2 mt-1 rounded-lg bg-[#121212] focus-within:ring-2 focus-within:ring-yellow-400 gap-2"
        onClick={() => document.getElementById("synonym-input").focus()}
      >
        {productLocalSynonyms.map((synonym, index) => (
          <div
            key={index}
            className="flex items-center px-2 py-1 text-sm bg-yellow-400 rounded-full text-primary"
          >
            {synonym}
            <button
              type="button"
              onClick={() =>
                setProductsLocalSynonyms((prev) =>
                  prev.filter((_, i) => i !== index)
                )
              }
              className="ml-2 text-primary hover:text-red-600"
            >
              X
            </button>
          </div>
        ))}
        <input
          id="synonym-input"
          type="text"
          className="flex-1 p-0 border-none bg-transparent focus:ring-0"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value.replace(/ /g, ""))} // Prevent spaces in input
          onKeyDown={handleKeyDown}
          placeholder={
            productLocalSynonyms.length === 0
              ? "Ex: Bateria Pilha Carregador"
              : ""
          }
        />
      </div>
      {/* Hidden input for form submission */}
      <input
        type="hidden"
        name="productSynonyms"
        value={JSON.stringify(productLocalSynonyms)}
      />
    </div>
  );
};

export default SynonymDetails;
