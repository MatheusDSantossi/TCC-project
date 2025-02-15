import { useEffect, useState } from "react";
import logo from "../assets/logo.svg";

import { Dialog } from "@mui/material";
import { DialogTitle } from "@mui/material";
import { DialogContent } from "@mui/material";

import ReactiveButton from "reactive-button";
import SynonymDetails from "./SynonymsDetails";
import AxiosInstance from "./AxiosInstance";

function ProductEdit({ open, onClose, product, onSave }) {
  const [updateButtonState, setUpdateButtonState] = useState("idle");
  const [deleteButtonState, setDeleteButtonState] = useState("idle");
  const [resetKey, setResetKey] = useState(0);
  const [productSynonyms, setProductSynonyms] = useState([]);

  // TODO: Add a delete fuction. The same screen of Register product

  const [formData, setFormData] = useState({
    productId: 0,
    productName: "",
    productBrand: "",
    productQuantityFormat: "",
    productSection: 0,
    productPrice: "",
    productMetricUnity: "",
    productAvailability: "",
    productSynonyms: [],
  });

  // useEffect(() => {
  //   setProductSynonyms(product?.synonymous || []);
  //   // setProductSynonyms([]);

  // }, [product]);

  useEffect(() => {
    console.log("Received product prop: ", product);
    // setButtonState("Carregando...");
    if (product) {
      console.log("Received product prop: ", product.brand);
      setFormData({
        productId: product.product_id || 0,
        productName: product.name || "",
        productBrand: product.brand || "",
        productQuantityFormat: product.quantity || "",
        productSection: product.secao_produto || 0,
        productPrice: product.preco_venda,
        productMetricUnity: product.unidade_medida,
        productAvailability: product.disponivel,
        productSynonyms: product.synonymous,
      });
      setResetKey((prev) => prev + 1);
      // setButtonState("Sucesso");
    }
  }, [product]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
    onClose();
  };

  const handleDeleteProduct = () => {
    onClose();
    try {
      AxiosInstance.delete(`delete-product/${formData.productId}/`);
      
      console.log(`Product ${formData.productName} deleted succesfuly`)
    } catch (error) {
      console.error("Error deleting product: ", error);
    }
    setResetKey((prev) => prev + 1);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      className="flex items-center justify-center min-h-screen p-4 bg-tertiary"
    >
      <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
        <div className="flex items-center justify-center mb-5">
          <img className="w-20 mr-4" src={logo} alt="" />
          <DialogTitle className="mb-6 text-2xl font-semibold text-center text-white left-40">
            Atualização de Produtos
          </DialogTitle>
        </div>
        <DialogContent>
          <form onSubmit={handleSubmit}>
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
                className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={formData?.productName}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    productName: e.target.value,
                  }))
                }
              />
            </div>
            <div className="grid grid-cols-2 gap-6">
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
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productBrand}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productBrand: e.target.value,
                    }))
                  }
                />
              </div>

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
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productQuantityFormat}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productQuantityFormat: e.target.value,
                    }))
                  }
                />
              </div>

              <div>
                <label
                  htmlFor="productSection"
                  className="block text-sm font-semibold text-white"
                >
                  Seção do Produto
                </label>
                <input
                  type="text"
                  id="productSection"
                  name="productSection"
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productSection}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productSection: e.target.value,
                    }))
                  }
                />
              </div>
              {/* ProductPrice */}
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
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productPrice}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productPrice: e.target.value,
                    }))
                  }
                />
              </div>
              {/* Product Metric Unity */}
              <div className="mb-5">
                <label
                  htmlFor="productMetricUnity"
                  className="block text-sm font-semibold text-white"
                >
                  Unidade de Medida do Produto
                </label>
                <select
                  type="text"
                  id="productMetricUnity"
                  name="productMetricUnity"
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productMetricUnity}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productMetricUnity: e.target.value,
                    }))
                  }
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
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productAvailability}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productAvailability: e.target.value,
                    }))
                  }
                />
              </div>
            </div>

            <div>
              <SynonymDetails
                setProductSynonyms={setProductSynonyms}
                resetKey={resetKey}
                previousProductSynonyms={formData?.productSynonyms}
                formData={formData}
                setFormData={setFormData}
                // onChange={(e) =>
                //   setFormData((prev) => ({
                //     ...prev,
                //     productSynonyms: e.target.value,
                //   }))
                // }
              />
            </div>
            <div className="flex">
              <ReactiveButton
                className="text-primary hover:bg-yellow-300"
                buttonState={updateButtonState}
                color="yellow"
                type={"submit"}
                loadingText="Carregando..."
                successText={"Atualizado com sucesso"}
                idleText={"Atualizar"}
                animation={true}
                rounded={true}
                width="120"
                height="40"
                size="large"
              />
              <ReactiveButton
                className="ml-20 text-primary hover:bg-yellow-300"
                buttonState={deleteButtonState}
                color="yellow"
                onClick={handleDeleteProduct}
                loadingText="Carregando..."
                successText={"Excluído com sucesso"}
                idleText={"Excluir"}
                animation={true}
                rounded={true}
                width="120"
                height="40"
                size="large"
              />
            </div>
          </form>
        </DialogContent>
      </div>
    </Dialog>
  );
}

export default ProductEdit;
