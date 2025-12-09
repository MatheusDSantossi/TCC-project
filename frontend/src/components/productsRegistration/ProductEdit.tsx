import { useEffect, useState } from "react";
import logo from "../../assets/logo.svg";

import { Dialog } from "@mui/material";
import { DialogTitle } from "@mui/material";
import { DialogContent } from "@mui/material";

import ReactiveButton from "reactive-button";
import SynonymDetails from "../SynonymsDetails";
import AxiosInstance from "../AxiosInstance";

import { MdOutlineClose } from "react-icons/md";

type productEditFormProps = {
  productId: number;
  productName: string;
  productBrand: string;
  productQuantityFormat: string;
  productSection: string;
  productPrice: string;
  productMetricUnity: string;
  productAvailability: string;
  productSynonyms: [];
  productLocalSynonyms?: [];
};

function ProductEdit({ open, onClose, product, onSave, refreshTable }) {
  const [updateButtonState, setUpdateButtonState] = useState("idle");
  const [deleteButtonState, setDeleteButtonState] = useState("idle");
  const [resetKey, setResetKey] = useState(0);
  const [productSynonyms, setProductSynonyms] = useState([]);

  const [formData, setFormData] = useState<productEditFormProps>({
    productId: 0,
    productName: "",
    productBrand: "",
    productQuantityFormat: "",
    productSection: "",
    productPrice: "",
    productMetricUnity: "",
    productAvailability: "",
    productSynonyms: [],
  });

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
        productSection: product.section || 0,
        productPrice: product.price,
        productMetricUnity: product.unit,
        productAvailability: product.stock,
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

  const handleDeleteProduct = async () => {
    onClose();
    try {
      const response = await AxiosInstance.delete(
        `products/${formData.productId}/`
        // `delete-product/${formData.productId}/`
      );

      refreshTable();

      console.log(
        `Product ${formData.productName} deleted successfully`,
        response.data
      );
    } catch (error) {
      console.error("Error deleting product: ", error);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      className="flex items-center justify-center min-h-screen p-1 bg-tertiary overflow-y-hidden"
    >
      <div className="w-full max-w-2xl p-8 rounded-lg shadow-xl bg-primary">
        <MdOutlineClose
          className="text-3xl text-white cursor-pointer hover:text-secondary"
          onClick={onClose}
        />
        <div className="flex items-center justify-center">
          <img className="w-20 mr-4" src={logo} alt="" />
          <DialogTitle className="text-2xl font-semibold text-center text-white ">
            Atualização de Produtos
          </DialogTitle>
        </div>
        <DialogContent>
          <form onSubmit={handleSubmit}>
            {/* # divide the product name into three parts (name, brand, quantity) */}
            {/* Product Name */}
            <div className="mb-6">
              <label
                htmlFor="productName"
                className="block mb-2 text-sm font-semibold text-white"
              >
                Nome do Produto
              </label>
              <input
                type="text"
                id="productName"
                name="productName"
                className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                value={formData?.productName}
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    productName: e.target.value,
                  }))
                }
              />
            </div>
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <label
                  htmlFor="productBrand"
                  className="block mb-2 text-sm font-semibold text-white"
                >
                  Marca do Produto
                </label>
                <input
                  type="text"
                  id="productBrand"
                  name="productBrand"
                  className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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
                  className="block mb-2 text-sm font-semibold text-white"
                >
                  Formato Quantidade do Produto
                </label>
                <input
                  type="text"
                  id="productQuantityFormat"
                  name="productQuantityFormat"
                  className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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
                  className="block mb-2 text-sm font-semibold text-white"
                >
                  Seção do Produto
                </label>
                <input
                  type="text"
                  id="productSection"
                  name="productSection"
                  className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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
                  className="block mb-2 text-sm font-semibold text-white"
                >
                  Preço do Produto
                </label>
                <input
                  type="text"
                  id="productPrice"
                  name="productPrice"
                  className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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
              <div className="mt-[-13px]">
                <label
                  htmlFor="productMetricUnity"
                  className="block p-3 text-sm font-semibold text-white"
                >
                  Unidade de Medida do Produto
                </label>
                <select
                  // type="text"
                  id="productMetricUnity"
                  name="productMetricUnity"
                  className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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
                  className="block mb-2 text-sm font-semibold text-white"
                >
                  Produtos em Estoque
                </label>
                <input
                  type="text"
                  id="productAvailability"
                  name="productAvailability"
                  className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
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

            <div className="mb-6">
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
            {/* Buttons */}
            <div className="flex gap-4 mt-8 ml-8">
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
