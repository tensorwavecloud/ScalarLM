gotemplate
{{- define "scalarlm.fullname" -}}
{{- printf "%s" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.vllmname" -}}
{{- printf "%s-vllm" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.labels" -}}
app.kubernetes.io/name: {{ include "scalarlm.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.vllmlabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.vllmname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
