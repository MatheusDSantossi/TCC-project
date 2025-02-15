import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";
import { IconButton } from "@mui/material";
import { MdEdit } from "react-icons/md";
import { useState } from "react";
import AxiosInstance from "../../AxiosInstance";
import ProductEdit from "../../ProductEdit";

export default function ProductsTable({ products_data, onProductUpdate }) {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);

  const handleEditClick = (product) => {
    console.log("Product: ", product);
    setSelectedProduct(product);
    setEditModalOpen(true);
  };

  const handleSave = async (updatedProduct) => {
    try {
      const response = await AxiosInstance.put(
        `update-product/${updatedProduct.productId}/`,
        {
          productName: updatedProduct.productName,
          productBrand: updatedProduct.productBrand,
          productQuantityFormat: updatedProduct.productQuantityFormat,
          productSection: updatedProduct.productSection,
          productPrice: updatedProduct.productPrice,
          productMetricUnity: updatedProduct.productMetricUnity,
          productAvailability: updatedProduct.productAvailability,
          productLocalSynonyms: updatedProduct.productLocalSynonyms,
          // productSynonyms: updatedProduct.productSynonyms,
        }
      );
      console.log("Product updated successfully: ", updatedProduct);

      if (response.status === 201 || response.status === 200) {
        onProductUpdate();
      }
    } catch (error) {
      console.error("Error updating product: ", error);
    }
  };

  const handleClose = () => {
    // setLoading(false);
    setEditModalOpen(false);
  };

  const columns = [
    {
      field: "product_id",
      headerName: <strong>ID</strong>,
      width: 90,
    },
    {
      field: "name",
      headerName: <strong>Produto</strong>,
      width: 200,
    },
    {
      field: "brand",
      headerName: <strong>Marca</strong>,
      width: 150,
    },
    {
      field: "quantity",
      headerName: <strong>Formato Quantidade</strong>,
      width: 180,
    },
    {
      field: "secao_produto",
      headerName: <strong>Seção</strong>,
      width: 100,
    },
    {
      field: "preco_venda",
      headerName: <strong>Preço de venda</strong>,
      width: 140,
    },
    {
      field: "actions",
      headerName: <strong>Ações</strong>,
      width: 100,
      sortable: false,
      renderCell: (params) => (
        <IconButton
          onClick={() => handleEditClick(params.row)}
          size="small"
          color="primary"
        >
          <MdEdit color="#5D76D3" />
        </IconButton>
      ),
    },
  ];

  let rows = [];

  const products_dict = () => {
    for (const product of products_data) {
      rows.push(product);
    }

    return rows;
  };

  console.log("Product dict: ", products_dict()[0]);

  // const rows = [
  //     { id: 1, lastName: 'Snow', firstName: 'Jon', age: 14 },
  //     { id: 2, lastName: 'Lannister', firstName: 'Cersei', age: 31 },
  //     { id: 3, lastName: 'Lannister', firstName: 'Jaime', age: 31 },
  //     { id: 4, lastName: 'Stark', firstName: 'Arya', age: 11 },
  //     { id: 5, lastName: 'Targaryen', firstName: 'Daenerys', age: null },
  //     { id: 6, lastName: 'Melisandre', firstName: null, age: 150 },
  //     { id: 7, lastName: 'Clifford', firstName: 'Ferrara', age: 44 },
  //     { id: 8, lastName: 'Frances', firstName: 'Rossini', age: 36 },
  //     { id: 9, lastName: 'Roxie', firstName: 'Harvey', age: 65 },
  //   ];

  return (
    <div className="flex items-center justify-center w-full overflow-x-auto">
      <Box sx={{ height: 650, width: 970 }}>
        <DataGrid
          rows={products_data}
          columns={columns}
          getRowId={(row) => row.product_id}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 10,
              },
            },
          }}
          pageSizeOptions={[10]}
          disableRowSelectionOnClick
        />
        <ProductEdit
          open={editModalOpen}
          onClose={handleClose}
          product={selectedProduct}
          onSave={handleSave}
        />
        
      </Box>
    </div>
  );
}
