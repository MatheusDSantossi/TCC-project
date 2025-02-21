import { useEffect, useRef, useState } from "react";

import "../App.css";
import { BsEraserFill, BsInfoCircleFill } from "react-icons/bs";
// Component

const SynonymDetails = ({
  formData,
  setFormData,
  setProductSynonyms,
  resetKey,
  previousProductSynonyms,
}) => {
  const [productLocalSynonyms, setProductsLocalSynonyms] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef(null);
  const [changingSynonyms, setChangingSynonyms] = useState(false);

  useEffect(() => {
    // console.log("Form Data before: ", formData)
    if (formData) {
      setFormData((formData) => ({
        ...formData,
        productLocalSynonyms,
      }));
    }
    // console.log("Form Data after: ", formData)
  }, [formData, setFormData, productLocalSynonyms]);

  // Clear local state on resetKey change
  useEffect(() => {
    setProductsLocalSynonyms([]);
    setInputValue("");
  }, [resetKey]);

  useEffect(() => {
    setProductSynonyms(productLocalSynonyms);
  }, [productLocalSynonyms, setProductSynonyms]);

  useEffect(() => {
    console.log("Changing synonyms: ", changingSynonyms);
    if (!previousProductSynonyms && productLocalSynonyms) {
      setProductsLocalSynonyms(productLocalSynonyms);
    } else if (previousProductSynonyms && !changingSynonyms) {
      setProductsLocalSynonyms(previousProductSynonyms);
    }
    console.log("Previous Synonyms: ", previousProductSynonyms);
  }, [
    productLocalSynonyms,
    setProductSynonyms,
    previousProductSynonyms,
    changingSynonyms,
  ]);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [productLocalSynonyms]);

  const handleKeyUp = (e) => {
    if (e.key === "   ") {
      const trimmed = inputValue.trim();

      if (trimmed) {
        if (!productLocalSynonyms.includes(trimmed)) {
          setProductsLocalSynonyms((prev) => [...prev, trimmed]);
        }
        setInputValue("");
      }
    }
    if (e.key === "Backspace" && inputValue === "") {
      setProductsLocalSynonyms((prev) => prev.slice(0, -1));
    }
  };

  const handleBlur = () => {
    const trimmed = inputValue.trim();

    if (trimmed) {
      if (!productLocalSynonyms.includes(trimmed)) {
        setProductsLocalSynonyms((prev) => [...prev, trimmed]);
      }
      setInputValue("");
    }
  };

  const clearSynonyms = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setChangingSynonyms(true);
    setProductsLocalSynonyms([]);
    setProductsLocalSynonyms([]);
  };

  return (
    <div className="mb-5">
      <div className="flex items-center justify-between w-full mb-1">
        <label
          htmlFor="productSynonyms"
          className="block text-sm font-semibold text-white"
        >
          Sinônimos para o Produto
        </label>
        <span className="p-1 rounded-full text-secondary">
          <BsInfoCircleFill />
        </span>

        <button
          className="p-1.5 rounded text-primary bg-secondary
      hover:bg-secondary-dark transition-colors"
          onClick={clearSynonyms}
        >
          <BsEraserFill />
        </button>
      </div>
      <div
        className="flex flex-wrap items-center w-full p-2 mt-1 rounded-lg bg-[#121212] focus-within:ring-2 focus-within:ring-yellow-400 gap-2"
        style={{
          maxHeight: "150px",
          overflow: "auto",
        }}
      >
        {productLocalSynonyms.map((synonym, index) => (
          <div
            key={index}
            className="flex items-center px-2 py-1 text-sm bg-yellow-400 rounded-full text-primary shrink-0"
          >
            {synonym}
            <button
              type="button"
              onClick={() => {
                setProductsLocalSynonyms((prev) =>
                  prev.filter((_, i) => i !== index)
                );
                setChangingSynonyms(true);
              }}
              className="ml-2 text-primary hover:text-red-600"
            >
              X
            </button>
          </div>
        ))}
        <input
          ref={inputRef}
          id="synonym-input"
          type="text"
          className="flex-1 p-0 bg-transparent border-none min-w-[120px] focus:ring-0 text-white"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
            setChangingSynonyms(true);
          }} // Prevent spaces in input
          // onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          onBlur={handleBlur}
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
