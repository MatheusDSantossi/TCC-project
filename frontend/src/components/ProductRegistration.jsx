import { useEffect, useState } from "react";
import logo from "../assets/logo.svg";
import AxiosInstance from "./AxiosInstance";

import ReactiveButton from "reactive-button";
import SynonymDetails from "./SynonymsDetails";

import InfoDialogScreen from "./InfoDialogScreen";

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

  // TODO: The process of getting the list of possible products is taking a way too long time.
  
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
      const numberOfSection = selectedProduct.section.slice(-1);
      console.log("Selected product in PREGISTER: ", selectedProduct)
      console.log("Number of section: " + numberOfSection);
      setProductBrand(selectedProduct.brand);
      setProductQuantityFormat(selectedProduct.quantity);
      setProductSection(numberOfSection);
      setProductPrice(selectedProduct.price);
      setProductMetricUnity(selectedProduct.unit);
      setProductAvailibility(selectedProduct.stock);
      setProductSynonyms(selectedProduct.synonymous);
    }
  };

  const registerProduct = (e) => {
    e.preventDefault();

    setButtonState("loading");

    console.log("Product synonyms: ", productSynonyms);

    AxiosInstance.post(`api/products/`, {
      name: productName.trim(),
      brand: productBrand.trim(),
      quantity: productQuantityFormat.trim(),
      section: Number(productSection),
      price: productPrice,
      unit: productMetricUnity.trim(),
      stock: productAvailability,
      synonymous: productSynonyms,
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
        setResetKey((prev) => prev + 1);
      })
      .catch((error) => {
        setButtonState("error");
        console.error("Error registering the product: ", error);
      });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-tertiary">
      <div className="w-full max-w-4xl p-8 rounded-lg shadow-xl bg-primary">
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
            <div className="flex items-center">
              <label
                htmlFor="productName"
                className="block text-lg font-semibold text-white"
              >
                Nome do Produto
              </label>
              {/* <span className="p-1 rounded-full text-secondary">
                <BsInfoCircleFill />
              </span> */}
              <InfoDialogScreen
                infoText={
                  <h2>
                    Coloque quatro letras ou mais para receber exemplos de produtos já
                    cadastrados no sistema. Com produto selecionado aperte a
                    tecla <span className="font-semibold">tab</span> para adicionar os dados do
                    determinado produto.
                  </h2>
                }
              />
            </div>
            <input
              type="text"
              id="productName"
              name="productName"
              placeholder="Digite o nome do produto"
              required
              className="w-full p-2 mt-1 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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

          <div className="grid grid-cols-2 gap-6 md:grid-cols-3">
            {/* Product brand */}
            <div className="col-span-1">
              <div className="flex items-center mb-5">
                <label
                  htmlFor="productBrand"
                  className="block text-lg font-semibold text-white"
                >
                  Marca do Produto
                </label>
                <InfoDialogScreen
                  infoText={
                    <h2>
                      Coloque a marca do produto preferencialmente texto
                      maiúsculo.
                    </h2>
                  }
                />
              </div>
              <input
                type="text"
                id="productBrand"
                name="productBrand"
                required
                className="w-full p-2 mt-1 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productBrand}
                onChange={(e) => setProductBrand(e.target.value)}
                placeholder="NESTLE"
              />
            </div>

            {/* Product Quantity Format */}
            <div>
              <div className="flex items-center">
                <label
                  htmlFor="productQuantityFormat"
                  className="block text-lg font-semibold text-white"
                >
                  Formato Quantidade do Produto
                </label>
                <InfoDialogScreen
                  infoText={
                    <h2>
                      Informe a quantidade de unidades por embalagem e a medida
                      de cada unidade. Exemplo: Se um fardo tem 6 garrafas de
                      2L, digite <span className="font-semibold">6X2L</span>
                    </h2>
                  }
                />
              </div>
              <input
                type="text"
                id="productQuantityFormat"
                name="productQuantityFormat"
                required
                className="w-full p-2 mt-1 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productQuantityFormat}
                onChange={(e) => setProductQuantityFormat(e.target.value)}
                placeholder="24X200G"
              />
            </div>

            {/* Product Section */}
            <div>
              <div className="flex items-center mb-5">
                <label
                  htmlFor="productSection"
                  className="block text-lg font-semibold text-white"
                >
                  Seção do Produto
                </label>
                <InfoDialogScreen
                  infoText={
                    <h2>
                      Informe o número da seção que o produto será colocado.
                    </h2>
                  }
                />
              </div>
              <input
                type="number"
                id="productSection"
                name="productSection"
                required
                className="w-full p-2 mt-1 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productSection}
                onChange={(e) => setProductSection(e.target.value)}
                placeholder="Seção 1"
              />
            </div>

            {/* Product Price */}
            <div>
              <div className="flex items-center">
                <label
                  htmlFor="productPrice"
                  className="block text-lg font-semibold text-white"
                >
                  Preço do Produto (R$)
                </label>
                <InfoDialogScreen
                  infoText={
                    <h2>
                      Informe o preço do produto (use ponto como separador
                      decimal). Exemplos: <span className="font-semibold">2.50</span> |{" "}
                      <span className="font-semibold">0.60</span> | <span className="font-semibold">10.99</span>
                    </h2>
                  }
                />
              </div>
              <input
                type="text"
                id="productPrice"
                name="productPrice"
                placeholder="10.99"
                required
                className="w-full p-2 mt-6 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productPrice}
                onChange={(e) => setProductPrice(e.target.value)}
              />
            </div>
            {/* Product Metric Unity */}
            <div>
              <div className="flex items-center">
                <label
                  htmlFor="productMetricUnity"
                  className="block text-lg font-semibold text-white"
                >
                  Unidade de Medida do Produto
                </label>
                <InfoDialogScreen
                  infoText={
                    <h2>
                      Selecione a unidade de medida correspondente à embalagem:
                      <br></br>
                      Exemplos:<br></br>
                      <span className="font-semibold">6X2L</span> → <span className="font-semibold">FD</span> (Fardo) |
                      <span className="font-semibold"> 12X200G </span> → <span className="font-semibold">CX</span> (Caixa) |
                      <span className="font-semibold"> 1X5KG</span> → <span className="font-semibold">UN</span> (Unidade)
                    </h2>
                  }
                />
              </div>
              <select
                id="productMetricUnity"
                name="productMetricUnity"
                required
                className="w-full p-[10px] mt-1 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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
              <div className="flex items-center mb-5">
                <label
                  htmlFor="productAvailability"
                  className="block text-lg font-semibold text-white"
                >
                  Produtos em Estoque
                </label>
                <InfoDialogScreen
                  infoText={
                    <h2>
                      Informe a quantidade de embalagens:<br></br>
                      Digite apenas o número total (ex: para 5 caixas com 12
                      unidades cada, insira <span className="font-semibold">5</span>)
                    </h2>
                  }
                />
              </div>
              <input
                type="text"
                id="productAvailability"
                name="productAvailability"
                required
                className="w-full p-2 mt-1 mb-5 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={productAvailability}
                onChange={(e) => setProductAvailibility(e.target.value)}
                placeholder="100"
              />
            </div>
          </div>
          <div>
            <SynonymDetails
              setProductSynonyms={setProductSynonyms}
              resetKey={resetKey}
              previousProductSynonyms={productSynonyms}
              // formData={formData}
              // setFormData={setFormData}
            />
          </div>

          <ReactiveButton
            className="text-primary hover:bg-yellow-300"
            buttonState={buttonState}
            color="yellow"
            type={"submit"}
            loadingText="Carregando..."
            successText={"Cadastrado com sucesso"}
            idleText={"Cadastrar"}
            animation="true"
            rounded="true"
            width={120}
            height={40}
            size="large"
            // onClick={registerProduct}
          />
        </form>
      </div>
    </div>
  );
}

export default ProductRegistration;
