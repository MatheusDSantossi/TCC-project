import { useCallback, useEffect, useReducer, useState } from "react";
import logo from "../../../assets/logo.svg";
import AxiosInstance from "../../AxiosInstance";
import ReactiveButton from "reactive-button";
import SynonymDetails from "../../SynonymsDetails";
import InfoDialogScreen from "../../InfoDialogScreen";

type Suggestion = { name: string; [k: string]: any };

type ButtonState = "idle" | "loading" | "success" | "error";

export type ProductState = {
  productName: string;
  productBrand: string;
  productQuantityFormat: string;
  productSection: string;
  productPrice: string;
  productMetricUnity: string;
  productAvailability: string;
  productSynonyms: string[]; // array of synonyms
  resetKey: number; // used to force child resets
  suggestions: Suggestion[];
  buttonState: ButtonState;
};

export const initialProductState: ProductState = {
  productName: "",
  productBrand: "",
  productQuantityFormat: "",
  productSection: "",
  productPrice: "",
  productMetricUnity: "",
  productAvailability: "",
  productSynonyms: [],
  resetKey: 0,
  suggestions: [],
  buttonState: "idle",
};

export type ProductAction =
  | { type: "SET_PRODUCT_NAME"; payload: string }
  | { type: "SET_PRODUCT_BRAND"; payload: string }
  | { type: "SET_PRODUCT_QUANTITY_FORMAT"; payload: string }
  | { type: "SET_PRODUCT_SECTION"; payload: string }
  | { type: "SET_PRODUCT_PRICE"; payload: string }
  | { type: "SET_PRODUCT_METRIC_UNITY"; payload: string }
  | { type: "SET_PRODUCT_AVAILABILITY"; payload: string }
  | { type: "SET_PRODUCT_SYNONYMS"; payload: string[] }
  | { type: "ADD_PRODUCT_SYNONYM"; payload: string } // push single
  | { type: "REMOVE_PRODUCT_SYNONYM"; payload: string } // remove by value
  | { type: "SET_SUGGESTIONS"; payload: Suggestion[] }
  | { type: "SET_BUTTON_STATE"; payload: ButtonState }
  | { type: "INCREMENT_RESET_KEY" }
  | { type: "RESET_FORM" };

// Reducer
export function productReducer(
  state: ProductState,
  action: ProductAction
): ProductState {
  switch (action.type) {
    case "SET_PRODUCT_NAME":
      return { ...state, productName: action.payload };
    case "SET_PRODUCT_BRAND":
      return { ...state, productBrand: action.payload };
    case "SET_PRODUCT_QUANTITY_FORMAT":
      return { ...state, productQuantityFormat: action.payload };
    case "SET_PRODUCT_SECTION":
      return { ...state, productSection: action.payload };
    case "SET_PRODUCT_PRICE":
      return { ...state, productPrice: action.payload };
    case "SET_PRODUCT_METRIC_UNITY":
      return { ...state, productMetricUnity: action.payload };
    case "SET_PRODUCT_AVAILABILITY":
      return { ...state, productAvailability: action.payload };
    case "SET_PRODUCT_SYNONYMS":
      console.log("Reducer SET_PRODUCT_SYNONYMS payload:", action.payload);
      const arr = Array.isArray(action.payload)
        ? action.payload
        : [action.payload];
      const cleaned = arr
        .flat()
        .map((s: any) => (s == null ? "" : String(s).trim()))
        .filter((s: string) => s.length > 0);
      return { ...state, productSynonyms: cleaned };
    case "ADD_PRODUCT_SYNONYM":
      // accept both single string or array to be robust
      const incoming = Array.isArray(action.payload)
        ? action.payload.flat()
        : [action.payload];
      const cleanedIncoming = incoming
        .map((s: any) => (s == null ? "" : String(s).trim()))
        .filter((s: string) => s.length > 0);

      if (cleanedIncoming.length === 0) return state;

      // merge avoiding exact duplicates
      const next = Array.from(
        new Set([...state.productSynonyms, ...cleanedIncoming])
      );
      return { ...state, productSynonyms: next };

    case "REMOVE_PRODUCT_SYNONYM":
      const toRemove = String(action.payload ?? "").trim();
      if (!toRemove) return state;
      return {
        ...state,
        productSynonyms: state.productSynonyms.filter((s) => s !== toRemove),
      };
    case "SET_SUGGESTIONS":
      return { ...state, suggestions: action.payload };
    case "SET_BUTTON_STATE":
      return { ...state, buttonState: action.payload };
    case "INCREMENT_RESET_KEY":
      return { ...state, resetKey: state.resetKey + 1 };
    case "RESET_FORM":
      // keep incremented resetKey so consumers depending on it will remount
      return { ...initialProductState, resetKey: state.resetKey + 1 };
    default:
      return state;
  }
}

const TabManual = () => {
  const [state, dispatch] = useReducer(productReducer, initialProductState);

  // Destructure for easy usage
  const {
    productName,
    productBrand,
    productQuantityFormat,
    productSection,
    productPrice,
    productMetricUnity,
    productAvailability,
    productSynonyms,
    suggestions,
    buttonState,
    resetKey,
  } = state;

  const onNameChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    dispatch({ type: "SET_PRODUCT_NAME", payload: e.target.value });
  }, []);
  const onBrandChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      dispatch({ type: "SET_PRODUCT_BRAND", payload: e.target.value });
    },
    []
  );
  const onProductAvailabilityChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      dispatch({ type: "SET_PRODUCT_AVAILABILITY", payload: e.target.value });
    },
    []
  );
  const onMetricUnityChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      dispatch({ type: "SET_PRODUCT_METRIC_UNITY", payload: e.target.value });
    },
    []
  );
  const onPriceChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      dispatch({ type: "SET_PRODUCT_PRICE", payload: e.target.value });
    },
    []
  );
  const onQuantityFormatChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      dispatch({
        type: "SET_PRODUCT_QUANTITY_FORMAT",
        payload: e.target.value,
      });
    },
    []
  );
  const onSectionChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      dispatch({ type: "SET_PRODUCT_SECTION", payload: e.target.value });
    },
    []
  );

  const onAddSynonym = useCallback((syn: string) => {
    dispatch({ type: "ADD_PRODUCT_SYNONYM", payload: syn });
  }, []);

  // TODO: The process of getting the list of possible products is taking a way too long time.

  const GetProductList = (productName) => {
    if (productName.length > 3) {
      AxiosInstance.get(`api/products/?search=${productName}`)
        .then((response) => {
          console.log(response.data);
          dispatch({ type: "SET_SUGGESTIONS", payload: response.data });
          // setSuggestions(response.data);
        })
        .catch((err) => {
          console.error("Error fetching product suggestions: ", err);
          dispatch({ type: "SET_SUGGESTIONS", payload: [] });
        });
    } else {
      dispatch({ type: "SET_SUGGESTIONS", payload: [] });
    }
  };

  useEffect(() => {
    GetProductList(productName);
  }, [productName]);

  const handleProductSelection = (selectedProductName) => {
    const selectedProduct = suggestions.find(
      (product) => product.name === selectedProductName
    );

    if (!selectedProduct) return;

    const numberOfSection = selectedProduct.section.slice(-1);
    console.log("Selected product in PREGISTER: ", selectedProduct);
    console.log("Number of section: " + numberOfSection);
    dispatch({
      type: "SET_PRODUCT_BRAND",
      payload: selectedProduct.brand ?? "",
    });
    dispatch({
      type: "SET_PRODUCT_QUANTITY_FORMAT",
      payload: selectedProduct.quantity ?? "",
    });
    dispatch({ type: "SET_PRODUCT_SECTION", payload: numberOfSection ?? "" });
    dispatch({
      type: "SET_PRODUCT_PRICE",
      payload: String(selectedProduct.price ?? ""),
    });
    dispatch({
      type: "SET_PRODUCT_METRIC_UNITY",
      payload: selectedProduct.unit ?? "",
    });
    dispatch({
      type: "SET_PRODUCT_AVAILABILITY",
      payload: selectedProduct.stock ?? "",
    });

    // === Sanitizing synonyms before dispatching ===
    const rawSyn = selectedProduct.synonymous ?? selectedProduct.synonyms ?? [];
    const normalizedSynonyms: string[] = Array.isArray(rawSyn)
      ? rawSyn
          .flat() // flatten nested arrays just in case
          .map((s) => (s == null ? "" : String(s).trim()))
          .filter((s) => s.length > 0)
      : String(rawSyn).trim().length
      ? [String(rawSyn).trim()]
      : [];

    // We only dispatch if we actually have at least one non-empty synonym
    if (normalizedSynonyms.length > 0) {
      dispatch({ type: "SET_PRODUCT_SYNONYMS", payload: normalizedSynonyms });
    } else {
      // clear synonyms when product has none
      dispatch({ type: "SET_PRODUCT_SYNONYMS", payload: [] });
    }

    console.log("normalizedSynonyms:", normalizedSynonyms);
  };

  const registerProduct = (e) => {
    e.preventDefault();

    dispatch({ type: "SET_BUTTON_STATE", payload: "loading" });

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
          // setButtonState("success");
          dispatch({ type: "SET_BUTTON_STATE", payload: "success" });
        }, 2000);

        console.log("Response: ", response);

        dispatch({ type: "RESET_FORM" });
      })
      .catch((error) => {
        dispatch({ type: "SET_BUTTON_STATE", payload: "error" });
        console.error("Error registering the product: ", error);
      });
  };
  return (
    <div className="w-full md:w-[570px] xl:w-4xl md:h-[660px] xl:h-[630px] max-w-4xl p-8 rounded-lg shadow-xl bg-primary  transition-all duration-300 ease">
      <div className="flex items-center mb-7">
        <img className="w-20 mr-4" src={logo} alt="Hecate's logo" />
        <h2 className="mb-6 m-auto text-2xl font-semibold text-center left-40">
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
                  Coloque quatro letras ou mais para receber exemplos de
                  produtos já cadastrados no sistema. Com produto selecionado
                  aperte a tecla <span className="font-semibold">tab</span> para
                  adicionar os dados do determinado produto.
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
            autoFocus
            onChange={(e) => {
              onNameChange(e);

              handleProductSelection(e.target.value);
            }}
            // onChange={(e) => setProductName(e.target.value)}
            // onBlur={(e) => handleProductSelection(e.target.value)} // Triggered when a product is selected
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
              onChange={onBrandChange}
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
                    Informe a quantidade de unidades por embalagem e a medida de
                    cada unidade. Exemplo: Se um fardo tem 6 garrafas de 2L,
                    digite <span className="font-semibold">6X2L</span>
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
              onChange={onQuantityFormatChange}
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
              onChange={onSectionChange}
              placeholder="Seção 1"
            />
          </div>

          {/* Product Price */}
          <div>
            <div className="flex items-center">
              <label
                htmlFor="productPrice"
                className="block text-lg font-semibold text-white -mb-5"
              >
                Preço do Produto (R$)
              </label>
              <InfoDialogScreen
                infoText={
                  <h2>
                    Informe o preço do produto (use ponto como separador
                    decimal). Exemplos:{" "}
                    <span className="font-semibold">2.50</span> |{" "}
                    <span className="font-semibold">0.60</span> |{" "}
                    <span className="font-semibold">10.99</span>
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
              onChange={onPriceChange}
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
                    <span className="font-semibold">6X2L</span> →{" "}
                    <span className="font-semibold">FD</span> (Fardo) |
                    <span className="font-semibold"> 12X200G </span> →{" "}
                    <span className="font-semibold">CX</span> (Caixa) |
                    <span className="font-semibold"> 1X5KG</span> →{" "}
                    <span className="font-semibold">UN</span> (Unidade)
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
              onChange={onMetricUnityChange}
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
                    unidades cada, insira{" "}
                    <span className="font-semibold">5</span>)
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
              onChange={onProductAvailabilityChange}
              placeholder="100"
            />
          </div>
        </div>
        <div>
          <SynonymDetails
            setProductSynonyms={onAddSynonym}
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
          animation={true}
          rounded={true}
          width={"120"}
          height={"40"}
          size="large"
          // onClick={registerProduct}
        />
      </form>
    </div>
  );
};

export default TabManual;
