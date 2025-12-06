import Box from "@mui/material/Box";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { IconButton } from "@mui/material";
import { MdEdit } from "react-icons/md";
import { useMemo, useState } from "react";
import AxiosInstance from "../../AxiosInstance";
import ProductEdit from "../../productsRegistration/ProductEdit";
import CustomLoadingOverlay, {
  CustomNoRowsOverlay,
} from "./components/CustomLoadingOverlay";

export default function ProductsTable({
  products_data,
  onProductUpdate,
  isLoading,
}) {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);

  const handleEditClick = (product) => {
    // console.log("Product: ", product);
    setSelectedProduct(product);
    setEditModalOpen(true);
  };

  const handleSave = async (updatedProduct) => {
    try {
      const response = await AxiosInstance.put(
        `api/products/${updatedProduct.productId}/`,
        {
          name: updatedProduct.productName,
          brand: updatedProduct.productBrand,
          quantity: updatedProduct.productQuantityFormat,
          section: updatedProduct.productSection,
          price: updatedProduct.productPrice,
          unit: updatedProduct.productMetricUnity,
          stock: updatedProduct.productAvailability,
          synonymous: updatedProduct.productLocalSynonyms,
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

  const handleRefreshTable = () => {
    console.log("Entered to refresh table");
    onProductUpdate();
  };

  const handleClose = () => {
    // setLoading(false);
    setEditModalOpen(false);
  };

  const columns = useMemo(
    () => [
      {
        field: "product_id",
        headerName: <strong className="text-base">ID</strong>,
        width: 90,
        minWidth: 80,
      },
      {
        field: "name",
        headerName: <strong className="text-base">Produto</strong>,
        minWidth: 200,
        flex: 2,
        renderCell: (params) => (
          <div
            style={{
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {params.value}
          </div>
        ),
      },
      {
        field: "brand",
        headerName: <strong className="text-base">Marca</strong>,
        minWidth: 150,
        flex: 1,
        renderCell: (p) => (
          <div
            style={{
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {p.value}
          </div>
        ),
      },
      {
        field: "quantity",
        headerName: <strong className="text-base">Formato Qt.</strong>,
        minWidth: 180,
        flex: 1,
      },
      {
        field: "section",
        headerName: <strong className="text-base">Seção</strong>,
        minWidth: 100,
        flex: 0.5,
      },
      {
        field: "price",
        headerName: <strong className="text-base">Preço</strong>,
        width: 80,
        flex: 1,
      },
      {
        field: "actions",
        headerName: <strong className="text-base">Ações</strong>,
        width: 50,
        flex: 0.5,
        sortable: false,
        renderCell: (params) => (
          <IconButton
            className="items-center"
            onClick={() => handleEditClick(params.row)}
            size="small"
            color="primary"
          >
            <MdEdit style={{ color: "var(--color-quaternary)", alignItems: "center" }} />
          </IconButton>
        ),
      },
    ],
    []
  );

  // Memoize rows so they don't change identity every render
  const rows = useMemo(() => products_data, [products_data]);

  // console.log("Product dict: ", products_dict()[0]);

  return (
    <div className="flex w-full h-[689px]">
      <Box
        sx={{
          position: "relative",
          display: "flex",
          flexDirection: "column",
          width: "100%",
          pl: 1.5,
          pr: 1,
          py: 1,
          bottom: 0,
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 1,
          "& .MuiDataGrid-root": {
            minWidth: 100, // Minimum width before scrolling
          },
        }}
      >
        <DataGrid
          rows={rows}
          columns={columns}
          getRowId={(row) => row.product_id}
          components={{
            LoadingOverlay: CustomLoadingOverlay,
            NoRowsOverlay: CustomNoRowsOverlay,
          }}
          slots={{
            noRowsOverlay: CustomNoRowsOverlay,
          }}
          slotProps={{
            noRowsOverlay: {
              // pass a dynamic message based on your app state
              content: isLoading
                ? "Carregando..."
                : "Nenhum produto cadastrado.",
            },
          }}
          componentsProps={{
            noRowsOverlay: {
              // pass a dynamic message based on your app state
              message: isLoading
                ? "Carregando..."
                : "Nenhum produto cadastrado.",
            },
          }}
          loading={!!isLoading}
          // re-enable virtualization (remove disableVirtualization)
          disableSelectionOnClick
          // pageSizeOptions={[10, 15, 30]}
          initialState={{ pagination: { paginationModel: { pageSize: 15 } } }}
          density="comfortable"
          // important: set fixed row/header heights that match your styling
          rowHeight={28} // tune to match your desired visual height
          headerHeight={30}
          sx={{
            fontSize: "14px",
            boxShadow: 2,
            border: 2,
            borderColor: "var(--color-quaternary)",
            backgroundColor: "#FFFFFFFF",
            "& .MuiDataGrid-cell": {
              backgroundColor: "#F4F4F4FF",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
              lineHeight: "1.2",
              padding: "8px",
            },
            "& .MuiDataGrid-columnHeaderTitle": {
              fontWeight: "bold",
            },
            "& .MuiDataGrid-cell:hover": {
              color: "var(--color-quaternary)",
            },
          }}
        />
        <ProductEdit
          open={editModalOpen}
          onClose={handleClose}
          product={selectedProduct}
          onSave={handleSave}
          refreshTable={handleRefreshTable}
        />
      </Box>
    </div>
  );
}
