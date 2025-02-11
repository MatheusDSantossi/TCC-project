import { useEffect, useState } from "react";
import logo from "../assets/logo.svg";
import AxiosInstance from "./AxiosInstance";

import ReactiveButton from "reactive-button";
import SynonymDetails from "./SynonymsDetails";

function ProductRegistration() {
  const [productName, setProductName] = useState("");
  const [productBrand, setProductBrand] = useState("");
  const [productQuantityFormat, setProductQuantityFormat] = useState("");
  const [productSection, setProductSection] = useState(0);
  const [productPrice, setProductPrice] = useState("");
  const [productMetricUnity, setProductMetricUnity] = useState("");
  const [productAvailability, setProductAvailibility] = useState("");
  const [productSynonyms, setProductSynonyms] = useState([]);
  const [resetKey, setResetKey] = useState(0);

  const [suggestions, setSuggestions] = useState([]);

  const [buttonState, setButtonState] = useState("idle");

  const GetProductList = (productName) => {
    if (productName.length > 3) {
      AxiosInstance.get(`api/products/?search=${productName}`)
        .then((response) => {
          console.log(response.data);
          setSuggestions(response.data);
        })
        .catch((err) => {
          console.error("Error fetching product suggestions: ", err);
          setSuggestions([]);
        });
    } else {
      setSuggestions([]);
    }
  };

  useEffect(() => {
    GetProductList(productName);
  }, [productName]);

  const handleProductSelection = (selectedProductName) => {
    const selectedProduct = suggestions.find(
      (product) => product.name === selectedProductName
    );

    if (selectedProduct) {
      const numberOfSection = selectedProduct.secao_produto.slice(-1);

      console.log("Number of section: " + numberOfSection);
      setProductBrand(selectedProduct.brand);
      setProductQuantityFormat(selectedProduct.quantity);
      setProductSection(numberOfSection);
      setProductPrice(selectedProduct.preco_venda);
      setProductMetricUnity(selectedProduct.unidade_medida);
      setProductAvailibility(selectedProduct.disponivel);
    }
  };

  const registerProduct = (e) => {
    e.preventDefault();

    setButtonState("loading");

    console.log("Product synonyms: ", productSynonyms);

    AxiosInstance.post(`register-product/post/`, {
      productName: productName.trim(),
      productBrand: productBrand.trim(),
      productQuantityFormat: productQuantityFormat.trim(),
      productSection: Number(productSection),
      productPrice: productPrice,
      productMetricUnity: productMetricUnity.trim(),
      productAvailability: productAvailability,
      productSynonyms: productSynonyms,
    })
      .then((response) => {
        if (!response) {
          throw new Error("Failed to register the product");
        }

        setTimeout(() => {
          setButtonState("success");
        }, 2000);

        console.log("Response: ", response);
        setProductName("");
        setProductBrand("");
        setProductQuantityFormat("");
        setProductSection(0);
        setProductPrice("");
        setProductMetricUnity("");
        setProductAvailibility("");
        setProductSynonyms([]);
        setResetKey(prev => prev + 1);
      })
      .catch((error) => {
        setButtonState("error");
        console.error("Error registering the product: ", error);
      });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-tertiary">
      <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
        <div className="flex items-center justify-center mb-5">
          <img className="w-20 mr-4" src={logo} alt="" />
          <h2 className="mb-6 text-2xl font-semibold text-center left-40">
            Cadastro de Produtos
          </h2>
        </div>
        <form onSubmit={registerProduct}>
          {/* # divide the product name into three parts (name, brand, quantity) */}
          {/* Product Name */}
          <div className="mb-5">
            <label
              htmlFor="productName"
              className="block text-sm font-semibold text-white"
            >
              Nome do Produto
            </label>
            <input
              type="text"
              id="productName"
              name="productName"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productName}
              list="productsSuggestions"
              onChange={(e) => setProductName(e.target.value)}
              onBlur={(e) => handleProductSelection(e.target.value)} // Triggered when a product is selected
            />
            <datalist id="productsSuggestions">
              {suggestions.map((product, index) => (
                <option key={index} value={product.name}>
                  {product.name}
                </option>
              ))}
            </datalist>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Product brand */}
            <div>
              <label
                htmlFor="productBrand"
                className="block text-sm font-semibold text-white"
              >
                Marca do Produto
              </label>
              <input
                type="text"
                id="productBrand"
                name="productBrand"
                required
                className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productBrand}
                onChange={(e) => setProductBrand(e.target.value)}
                placeholder="Néscau"
              />
            </div>

            {/* Product Quantity Format */}
            <div>
              <label
                htmlFor="productQuantityFormat"
                className="block text-sm font-semibold text-white"
              >
                Formato Quantidade do Produto
              </label>
              <input
                type="text"
                id="productQuantityFormat"
                name="productQuantityFormat"
                required
                className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productQuantityFormat}
                onChange={(e) => setProductQuantityFormat(e.target.value)}
                placeholder="24X200G"
              />
            </div>

            {/* Product Section */}
            <div>
              <label
                htmlFor="productSection"
                className="block text-sm font-semibold text-white"
              >
                Seção do Produto
              </label>
              <input
                type="number"
                id="productSection"
                name="productSection"
                required
                className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productSection}
                onChange={(e) => setProductSection(e.target.value)}
                placeholder="Seção 1"
              />
            </div>

            {/* Product Price */}
            <div>
              <label
                htmlFor="productPrice"
                className="block text-sm font-semibold text-white"
              >
                Preço do Produto
              </label>
              <input
                type="text"
                id="productPrice"
                name="productPrice"
                required
                className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productPrice}
                onChange={(e) => setProductPrice(e.target.value)}
              />
            </div>
            {/* Prouct Metric Unity */}
            <div>
              <label
                htmlFor="productMetricUnity"
                className="block text-sm font-semibold text-white"
              >
                Unidade de Medida do Produto
              </label>
              <select
                id="productMetricUnity"
                name="productMetricUnity"
                required
                className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productMetricUnity}
                onChange={(e) => setProductMetricUnity(e.target.value)}
              >
                <option value="">Selecione uma opção</option>
                <option value="CX">CX</option>
                <option value="UN">UN</option>
                <option value="KG">KG</option>
                <option value="L">L</option>
                <option value="M">M</option>
                <option value="PC">PC</option>
                <option value="PT">PT</option>
                <option value="PCT">PCT</option>
                <option value="PR">PR</option>
              </select>
            </div>
            {/* Product Availability */}
            <div>
              <label
                htmlFor="productAvailability"
                className="block text-sm font-semibold text-white"
              >
                Produtos em Estoque
              </label>
              <input
                type="text"
                id="productAvailability"
                name="productAvailability"
                required
                className="w-full p-2 mt-1 mb-5 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productAvailability}
                onChange={(e) => setProductAvailibility(e.target.value)}
                placeholder="100"
              />
            </div>
          </div>
            <div>
              <SynonymDetails setProductSynonyms={setProductSynonyms} resetKey={resetKey} />
              {/* <input
                type="text"
                id="productSynonymous"
                name="productSynonymous"
                required
                className="w-full p-2 mt-1 mb-5 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productSynonyms}
                onChange={(e) => setproductSynonyms(e.target.value)}
                placeholder="Bateria, Pilha..."
              /> */}
            </div>

          {/* <button
            className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
            type="submit"
            onClick={registerProduct}
          >
            Register
          </button> */}
          <ReactiveButton
            className="text-primary hover:bg-yellow-300"
            buttonState={buttonState}
            color="yellow"
            type={"submit"}
            loadingText="Carregando..."
            successText={"Cadastrado com sucesso"}
            idleText={"Cadastrar produto"}
            animation={true}
            rounded={true}
            width="120"
            height="40"
            size="large"
            // onClick={registerProduct}
          />
        </form>
      </div>
    </div>
  );
}

export default ProductRegistration;
