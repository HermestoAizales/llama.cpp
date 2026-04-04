<script lang="ts">
	import { ChevronDown, ChevronUp } from '@lucide/svelte';

	export interface TokenLogprobData {
		token: string;
		logprob: number;
		top_logprobs: Array<{ token: string; logprob: number }>;
	}

	interface Props {
		tokens: TokenLogprobData[];
	}

	let { tokens }: Props = $props();
	let expanded = $state(false);

	function computeEntropy(data: TokenLogprobData): number {
		let h = 0;
		const items = data.top_logprobs.length > 0 ? data.top_logprobs : [{ token: data.token, logprob: data.logprob }];
		for (const t of items) {
			const p = Math.exp(t.logprob);
			if (p > 0 && p <= 1) {
				h -= p * Math.log2(p);
			}
		}
		return h;
	}

	function getEntropyColor(v: number): string {
		if (v < 1) return 'text-green-500';
		if (v < 3) return 'text-yellow-500';
		return 'text-red-500';
	}

	function probColor(p: number): string {
		if (p >= 0.7) return 'bg-green-500';
		if (p >= 0.3) return 'bg-yellow-500';
		return 'bg-red-500';
	}

	function formatToken(t: string): string {
		return t.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
	}

	let displayTokens = $derived(tokens.slice(0, 100));
	let hasMore = $derived(tokens.length > 100);
</script>

<div class="border rounded-md border-border bg-muted/30 mt-2">
	<button
		onclick={() => (expanded = !expanded)}
		class="flex w-full items-center justify-between px-3 py-2 text-sm hover:bg-muted/50 rounded-md"
	>
		<div class="flex items-center gap-2">
			{#if expanded}
				<ChevronDown class="h-4 w-4" />
			{:else}
				<ChevronUp class="h-4 w-4" />
			{/if}
			<span class="font-medium">Token Inspection</span>
			<span class="text-xs text-muted-foreground">
				{tokens.length} tokens
			</span>
		</div>
	</button>

	{#if expanded}
		<div class="px-3 pb-3 space-y-1 max-h-96 overflow-y-auto">
			{#each displayTokens as t, idx}
				{@const items = t.top_logprobs.length > 0 ? t.top_logprobs : [{ token: t.token, logprob: t.logprob }]}
				{@const mainProb = Math.exp(t.logprob)}
				{@const entropy = computeEntropy(t)}
				<div
					class="rounded px-2 py-1.5 text-xs border border-border/50 bg-background/50 hover:bg-muted/30 transition-colors cursor-default"
				>
					<div class="flex items-center gap-2 mb-1">
						<span class="text-muted-foreground font-mono font-bold select-none w-7 text-right">
							#{idx + 1}
						</span>
						<span class="font-mono font-semibold truncate max-w-[120px]" title={t.token}>
							{formatToken(t.token)}
						</span>
						<span class="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium text-white tabular-nums" class:probColor={mainProb >= 0.7}>
							{(mainProb * 100).toFixed(1)}%
						</span>
						<span class="text-muted-foreground tabular-nums ml-auto">
							logprob: {t.logprob.toFixed(3)}
						</span>
						<span class="tabular-nums font-medium" class:getEntropyColor={true}>
							H={entropy.toFixed(2)}
						</span>
					</div>

					{#if items.length > 1}
						<div class="space-y-0.5 mt-1 border-t border-border/30 pt-1">
							{#each items.slice(0, 5) as alt, ai}
								{@const altProb = Math.exp(alt.logprob)}
								<div class="flex items-center gap-1">
									<span class="text-muted-foreground w-3 text-right text-[10px]">{ai + 1}.</span>
									<span class="font-mono text-[11px] w-16 truncate" title={alt.token}>
										{formatToken(alt.token)}
									</span>
									<div class="flex-1 h-3 bg-muted/40 rounded overflow-hidden">
										<div
											class="h-full rounded-sm"
											class:bg-green-500={altProb >= 0.7}
											class:bg-yellow-500={altProb >= 0.3 && altProb < 0.7}
											class:bg-red-500={altProb < 0.3}
											style={"width: " + Math.min(altProb * 100, 100) + "%;"}
										></div>
									</div>
									<span class="text-muted-foreground tabular-nums text-[10px] w-10 text-right">
										{(altProb * 100).toFixed(1)}%
									</span>
								</div>
							{/each}
							{#if items.length > 5}
								<span class="text-[10px] text-muted-foreground block text-right">
									+{items.length - 5} more
								</span>
							{/if}
						</div>
					{/if}
				</div>
			{/each}
			{#if hasMore}
				<div class="text-center text-xs text-muted-foreground py-1">
					Showing first 100 of {tokens.length} tokens
				</div>
			{/if}
		</div>
	{/if}
</div>
